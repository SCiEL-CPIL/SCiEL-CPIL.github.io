import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    The current best strategy (score 1.0150) uses a local degree-based heuristic:
    Score = -deg(u) + beta * avg_neighbor_deg(u)
    
    This new strategy rethinks the approach by focusing on GLOBAL structure rather than
    just local degree. In Erdos-Renyi graphs, the Maximum Independent Set (MIS) tends
    to include many low-degree vertices that are "peripheral" to the main connected
    components or dense clusters.
    
    Key Insight:
    Vertices with high "centrality" (like Eigenvector Centrality) are typically hubs
    or part of dense cores. Selecting them is bad for MIS size because they block many
    neighbors. Conversely, vertices with LOW centrality are likely on the periphery,
    having fewer connections to other important parts of the graph.
    
    By scoring vertices based on their centrality (specifically, we want to pick
    vertices with LOW centrality), we aim to select vertices that are structurally
    "isolated" or "leaf-like" in the global sense, not just locally low-degree.
    
    Eigenvector Centrality (or similar spectral methods) captures the influence of a
    vertex in the network. For MIS, we want to avoid high-influence vertices.
    
    Implementation:
    1. Compute the adjacency matrix A.
    2. Estimate Eigenvector Centrality. Since n=200, we can use power iteration
       or numpy's linear algebra directly. 
       Actually, for a symmetric matrix, the principal eigenvector of A gives the
       eigenvector centrality.
       However, simple power iteration might converge slowly if the spectral gap is small.
       Alternatively, we can use the inverse of the Laplacian or just the eigenvector
       of the adjacency matrix.
       
       Let's use the principal eigenvector of the adjacency matrix.
       Let v be the eigenvector corresponding to the largest eigenvalue of A.
       The components v[i] represent the centrality of node i.
       We want to MINIMIZE centrality. So score = -v[i].
       
       But wait, standard MIS heuristics often combine degree and centrality.
       A vertex with very low degree but high centrality (rare in ER, but possible)
       might still be bad.
       
       Let's look at the "cost" of picking u again.
       If we pick u, we remove N(u).
       The "value" of the remaining graph decreases.
       
       Another perspective: 
       The greedy algorithm picks the highest score first.
       If we use -centrality, we pick the most peripheral nodes first.
       This is a known heuristic for MIS: "Low Centrality First".
       
       Let's try a combination of Local Degree (fast, robust) and Global Centrality.
       
       Score = alpha * (-deg) + beta * (-centrality)
       
       We need to normalize centrality to be comparable to degree.
       Degree is in [0, n]. Centrality (eigenvector) can be scaled arbitrarily.
       We can normalize the eigenvector to have mean 0 and std 1, or max 1.
       
       Let's compute the principal eigenvector of A.
       For n=200, np.linalg.eigh or power iteration is fast enough.
       
       Actually, there is a simpler proxy for "peripherality" that is very fast:
       The "k-core" decomposition or simply the distance from the center.
       
       Let's stick to Eigenvector Centrality as it is mathematically well-founded for
       identifying hubs.
       
       Steps:
       1. A = np.asarray(adjacency, dtype=float)
       2. Compute principal eigenvector v of A.
          Since A is symmetric, we can use np.linalg.eigh.
          eigh returns eigenvalues in ascending order. The last one is the largest.
          The corresponding eigenvector is the principal one.
          We take the absolute value or ensure it's positive (eigenvectors can be +/-).
          Actually, for a connected graph with positive entries, the principal eigenvector
          has all positive components by Perron-Frobenius.
          If the graph is disconnected, it might have zeros or mixed signs for components
          with zero spectral radius, but ER(200, p) is likely connected or has a giant component.
          To be safe, we can take the absolute value of the eigenvector components?
          No, if the graph is bipartite-ish, signs might flip. But A is non-negative.
          The principal eigenvector of a non-negative matrix has non-negative components.
          
       3. Normalize v. Let v_norm = v / max(v). Now v_norm is in [0, 1].
       
       4. Combine with degree.
          score = -deg + beta * (-v_norm)  ?
          Or score = -deg - beta * v_norm ?
          
          Let's think about the trade-off.
          Low degree is good.
          Low centrality is good.
          
          Current best: -deg + 0.6 * avg_neighbor_deg.
          Note: avg_neighbor_deg is correlated with centrality?
          If a node is connected to high-degree nodes, its avg_neighbor_deg is high.
          High centrality nodes are connected to high centrality nodes.
          So -centrality might be similar to -avg_neighbor_deg?
          
          However, centrality is a GLOBAL property. A node might have low degree but be
          connected to a hub, giving it moderate centrality.
          A node might have moderate degree but be in a sparse cluster, giving it low centrality.
          
          Let's try a pure centrality approach first, or a blend.
          
          Given the compute budget (120s), eigendecomposition of 200x200 is trivial (<0.1s).
          
          Let's try:
          score = -deg - 0.5 * v_norm
          
          Wait, if v_norm is high for hubs, -v_norm is low. So hubs get low scores.
          Leaves get high scores.
          This aligns with -deg.
          
          Is there a benefit over -deg?
          -deg only looks at 1-hop.
          Centrality looks at infinite hops (damped).
          In ER graphs, the degree distribution is Poisson.
          The "hubs" are just high-degree nodes.
          Their neighbors are also likely high-degree (assortativity is ~0 in ER, but
          high-degree nodes connect to high-degree nodes by chance of having many edges).
          
          Let's try a different global metric: **Laplacian Eigenvalue**.
          The Fiedler vector (2nd smallest eigenvector of Laplacian) indicates community structure.
          This might be overkill.
          
          Let's try **Inverse of Clustering Coefficient**? No, too local.
          
          Let's try **PageRank**.
          PageRank is essentially eigenvector centrality with a damping factor.
          It is more robust to disconnected components.
          PR(i) = (1-d)/n + d * sum_{j in N(i)} PR(j)/deg(j)
          
          We can solve the linear system for PageRank.
          (I - d * A^T D^{-1}) PR = (1-d)/n * 1
          
          For n=200, solving this linear system is fast.
          
          Let's implement PageRank.
          d = 0.85 (standard).
          
          Score = -deg - beta * PR_norm
          
          Why this might be better:
          PageRank identifies "important" nodes. In MIS, we want to avoid important nodes
          because they block many others.
          The degree term handles the immediate blocking cost.
          The PageRank term handles the "structural importance" which correlates with
          how much of the graph is "tied" to this node.
          
          Let's test this hypothesis.
          
          Implementation details:
          1. D = diag(deg)
          2. M = I - d * A.T @ inv(D)  (Note: A is symmetric, so A.T = A)
             But D might have zeros. Handle isolated nodes.
             If deg[i] == 0, the term is 0.
             We can use a safe division.
             
          3. b = (1-d)/n * ones(n)
          
          4. Solve M @ pr = b
          
          5. Normalize pr: pr_norm = pr / max(pr)
          
          6. Score = -deg - 0.5 * pr_norm
          
          Let's refine the combination.
          The current best uses avg_neighbor_deg.
          avg_neighbor_deg is a local approximation of the neighbor's importance.
          PageRank is a global propagation of importance.
          
          I will use PageRank as the secondary term.
          
          One concern: If the graph is disconnected, PageRank still works (teleportation).
          Eigenvector centrality might fail or be arbitrary for disconnected components.
          So PageRank is safer.
          
          Let's write the code.
    """
    start_time = time.time()
    
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # 1. Calculate degrees
    deg = adj.sum(axis=1)
    
    # 2. Calculate PageRank
    # PageRank equation: PR = (1-d)/n * 1 + d * M @ PR
    # => (I - d*M) @ PR = (1-d)/n * 1
    # M is the column-stochastic matrix of the graph.
    # M[i, j] = 1/deg(j) if (i, j) is an edge, else 0.
    # Note: adjacency is symmetric, but PageRank transition is typically from j to i.
    # If we define M such that M @ PR gives the flow, M[i,j] corresponds to edge j->i.
    # Since graph is undirected, A[i,j] = A[j,i].
    # M = A @ D^{-1} (if A is symmetric, A.T = A, so M = A @ D^{-1})
    # Wait, standard definition: M_{ij} = 1/deg(j) if i~j.
    # So M = A @ D^{-1}.
    
    d = 0.85
    safe_deg = np.where(deg == 0, 1.0, deg)
    # D_inv is a diagonal matrix, but we can just divide columns.
    # M = A / safe_deg[None, :]  (Broadcasting: divide each column j by deg[j])
    
    M = adj / safe_deg[None, :]
    
    # Construct the system matrix
    I = np.eye(n)
    A_mat = I - d * M
    b_vec = np.ones(n) * (1.0 - d) / n
    
    try:
        # Solve for PageRank
        pr = np.linalg.solve(A_mat, b_vec)
    except np.linalg.LinAlgError:
        # Fallback if singular (unlikely with teleportation)
        pr = np.ones(n) / n
        
    # Normalize PageRank for scoring
    # We want to penalize high PageRank (hubs)
    pr_max = np.max(pr)
    if pr_max > 0:
        pr_norm = pr / pr_max
    else:
        pr_norm = np.zeros(n)
        
    # 3. Combine scores
    # Primary: Minimize degree (local cost)
    # Secondary: Minimize PageRank (global structural cost)
    
    # Tuning beta:
    # In the current best, the secondary term was ~0.5 * avg_neighbor_deg.
    # avg_neighbor_deg is roughly deg * (assortativity) or just average degree.
    # PageRank is normalized to [0,1].
    # Degree is ~10.
    # So -deg is ~ -10.
    # -0.5 * pr_norm is ~ -0.5.
    # This seems reasonable.
    
    beta = 0.5
    score = -deg - beta * pr_norm
    
    # Safety deadline check
    if time.time() - start_time > 140:
        return -deg
        
    return score
