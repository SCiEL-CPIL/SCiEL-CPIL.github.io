import math
import time
import numpy as np
from scipy.optimize import minimize

def _jc_distance(a, b):
    """Calculate Jukes-Cantor corrected distance between two sequences."""
    if len(a) == 0:
        return 0.0
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / len(a)
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _build_site_patterns(sequences):
    """Group identical columns to speed up likelihood calculation."""
    labels = sorted(sequences.keys(), key=lambda x: int(x[1:]))
    n_taxa = len(labels)
    n_sites = len(sequences[labels[0]])
    
    # Map A,C,G,T to 0,1,2,3
    char_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    patterns = {}
    for i in range(n_sites):
        col = tuple(char_to_idx[sequences[label][i]] for label in labels)
        patterns[col] = patterns.get(col, 0) + 1
    
    # Convert to numpy arrays for vectorization
    # patterns is dict {tuple: count}
    pattern_keys = list(patterns.keys())
    pattern_counts = np.array([patterns[k] for k in pattern_keys])
    pattern_matrix = np.array(pattern_keys).T # Shape: (4, K)
    
    return labels, pattern_matrix, pattern_counts

def _calc_likelihood(tree_adj, branch_lengths, pattern_matrix, pattern_counts):
    """
    Calculate log-likelihood of the tree given branch lengths and site patterns.
    tree_adj: adjacency list {node: {neighbor: branch_length}}
    branch_lengths: dict {(u,v): length}
    pattern_matrix: (4, K) array
    pattern_counts: (K,) array
    """
    # Felsenstein's algorithm requires a rooted tree.
    # We root the tree arbitrarily at node 0 (or any node).
    # Since the tree is unrooted, the likelihood is invariant to the root position.
    
    n_nodes = len(tree_adj)
    # Root at 0
    root = 0
    parent = {root: -1}
    children = {i: [] for i in range(n_nodes)}
    
    # BFS to determine parent-child relationships
    stack = [root]
    visited = {root}
    while stack:
        u = stack.pop()
        for v, _ in tree_adj[u].items():
            if v not in visited:
                visited.add(v)
                parent[v] = u
                children[u].append(v)
                stack.append(v)
                
    # Post-order traversal to compute likelihoods
    # L[u] is a vector of length 4, probability of state at u given data in subtree u
    L = {i: np.zeros(4) for i in range(n_nodes)}
    
    def get_prob_matrix(d):
        # JC69 transition probabilities
        # P(x->y)
        # if x==y: 0.25 + 0.75 * exp(-4/3 * d)
        # if x!=y: 0.25 - 0.25/3 * exp(-4/3 * d)
        e = math.exp(-4.0/3.0 * d)
        p_same = 0.25 + 0.75 * e
        p_diff = 0.25 - 0.25/3.0 * e
        P = np.full((4,4), p_diff)
        for i in range(4):
            P[i,i] = p_same
        return P

    # Process nodes in reverse topological order (leaves first)
    # We can just iterate and ensure children are processed first
    # Since it's a tree, we can do a simple DFS post-order
    
    def post_order(u):
        for c in children[u]:
            post_order(c)
            # Combine child c into u
            # L[c] is the likelihood at child c
            # We need to propagate L[c] up to u using the branch length between u and c
            d = branch_lengths.get((u,c), branch_lengths.get((c,u), 0.0))
            P = get_prob_matrix(d)
            # L[u] += P @ L[c] ? No, it's a sum over states at c.
            # L[u][s] = sum_{t} P(s->t) * L[c][t]
            L[u] += P.T @ L[c] # Wait, P is s->t. L[c][t] is prob of t at c.
            # So L[u][s] = sum_t P[s,t] * L[c][t]
            # This is (P.T @ L[c])[s]?
            # Let's check dimensions.
            # L[c] is shape (4,).
            # P is shape (4,4). P[s,t] is prob s->t.
            # We want result[s] = sum_t P[s,t] * L[c][t].
            # This is matrix-vector product P @ L[c].
            L[u] += P @ L[c]
            
        if not children[u]:
            # Leaf node
            # L[u][s] is the probability of observing the data at this leaf given state s
            # The data is the site patterns.
            # For each pattern k, with count c_k, the probability of observing pattern k given state s at u
            # is the product over sites? No, patterns are grouped.
            # For a specific pattern (col of states for all taxa), we only care about the state at this leaf u.
            # Wait, Felsenstein's algorithm works on the whole tree.
            # At a leaf, the "likelihood" vector is the probability of the observed data at that leaf.
            # Since we grouped sites, we need to be careful.
            # Actually, standard Felsenstein assumes independent sites.
            # If we group sites, we can compute the likelihood for each pattern and raise to the power of count.
            # L_leaf[u][s] = prod_{k} (Prob(pattern_k | state_s))^count_k
            # This is only valid if the leaf's state determines the pattern? No.
            # The pattern is the column of states for ALL taxa.
            # So for a leaf u, the observed data is the state of u in that column.
            # If the pattern is (s0, s1, ..., sN-1), and u is taxon i, then the observed state is si.
            # So Prob(pattern_k | state_s at u) is 1 if s == si, else 0?
            # No, that's for the leaf's own data.
            # In Felsenstein, the leaf likelihood is:
            # L_leaf[u][s] = 1 if s == observed_state, else 0.
            # But we have multiple sites.
            # So L_leaf[u][s] = (1 if s==obs else 0) for each site.
            # With patterns:
            # For pattern k (column vector), let obs_u be the state of u in that column.
            # The contribution to the log-likelihood for this pattern, given state s at u, is:
            # If s == obs_u, it's 1 (log 0). If s != obs_u, it's 0 (log -inf).
            # This means the likelihood at a leaf is a one-hot vector.
            # But we are summing over sites.
            # So L_leaf[u][s] = sum_{k} count_k * (1 if s == obs_{u,k} else 0) ?
            # No, likelihood is a product.
            # L_leaf[u][s] = prod_{k} (1 if s == obs_{u,k} else 0)^count_k
            # This is 1 if s matches obs_{u,k} for ALL k with count_k > 0?
            # No, that's wrong. The sites are independent.
            # The likelihood of the data at leaf u given state s is:
            # prod_{sites j} P(data_j | s)
            # P(data_j | s) is 1 if data_j == s, 0 otherwise.
            # So L_leaf[u][s] is 1 if s is the state at u for ALL sites? Impossible.
            
            # Ah, I see the confusion.
            # In Felsenstein's algorithm, the likelihood at a leaf u for state s is:
            # L_u(s) = 1 if s is the observed state at u, 0 otherwise.
            # This is for a SINGLE site.
            # For multiple sites, we sum the log-likelihoods.
            # So we should calculate the log-likelihood for each site pattern separately and sum them?
            # No, the pruning algorithm computes the likelihood of the whole tree for a single site.
            # To handle multiple sites, we can:
            # 1. Run the algorithm for each site pattern, weighted by count.
            # 2. Or, modify the leaf initialization.
            
            # Let's modify the leaf initialization.
            # L_leaf[u][s] should be the likelihood of the data at u given state s.
            # Since sites are independent, Log L = sum_{sites} Log P(data_site | s).
            # P(data_site | s) is 1 if data_site == s, 0 otherwise.
            # So Log P is 0 if data_site == s, -inf otherwise.
            # This means if s != data_site for ANY site, the total likelihood is 0.
            # This implies that the state at the leaf MUST match the data at the leaf for all sites.
            # This is trivially true for the observed data.
            # But in the tree, the state at the leaf is fixed to the observed data.
            # So, for a leaf u, L_u(s) is 1 if s == observed_state_u, 0 otherwise.
            # But we have multiple sites.
            # For site 1, obs is s1. For site 2, obs is s2.
            # If we assume the leaf state is a single state for the whole sequence? No, that's not how phylogenetics works.
            # The leaf represents the sequence. The state at the leaf for site j is the character at position j.
            # So, for site j, the leaf likelihood is a one-hot vector for the character at j.
            # For the whole tree, the likelihood for site j is computed by pruning.
            # Then we sum the log-likelihoods over all sites j.
            
            # So, the algorithm should be:
            # For each site pattern k (representing count_k sites):
            #   Run Felsenstein pruning to get the likelihood vector at the root.
            #   Sum over root states to get the total likelihood for this pattern.
            #   Add count_k * log(likelihood) to the total log-likelihood.
            
            # This is expensive if K is large.
            # But K is small (<1000).
            # And we can vectorize the pruning step?
            # The pruning step involves matrix multiplications.
            # We can process all patterns at once?
            # L[u] would be a matrix of shape (4, K).
            # L[u][s, k] = likelihood of state s at u given pattern k.
            
            # Let's implement this vectorized version.
            pass
            
    # Re-implementing with vectorization over patterns
    
    # L[u] will be a numpy array of shape (4, K)
    L = {i: np.zeros((4, len(pattern_counts))) for i in range(n_nodes)}
    
    def get_prob_matrix_vec(d):
        # Returns (4,4) matrix
        e = math.exp(-4.0/3.0 * d)
        p_same = 0.25 + 0.75 * e
        p_diff = 0.25 - 0.25/3.0 * e
        P = np.full((4,4), p_diff)
        for i in range(4):
            P[i,i] = p_same
        return P

    def post_order_vec(u):
        for c in children[u]:
            post_order_vec(c)
            d = branch_lengths.get((u,c), branch_lengths.get((c,u), 0.0))
            P = get_prob_matrix_vec(d)
            # L[u] += P @ L[c]
            # L[c] is (4, K). P is (4,4).
            # We want L[u][s, k] += sum_t P[s,t] * L[c][t, k]
            # This is (P @ L[c])[s, k]
            L[u] += P @ L[c]
            
        if not children[u]:
            # Leaf
            # For each pattern k, the observed state at leaf u is pattern_matrix[u, k]
            # L[u][s, k] is 1 if s == pattern_matrix[u, k], else 0.
            # We can create this matrix efficiently.
            obs = pattern_matrix[u] # shape (K,)
            # L[u] is (4, K)
            # L[u][s, k] = 1 if s == obs[k]
            # We can use broadcasting or one-hot encoding.
            # Create a one-hot matrix for obs: (K, 4)
            one_hot = np.zeros((len(obs), 4))
            one_hot[np.arange(len(obs)), obs] = 1
            L[u] = one_hot.T # shape (4, K)
            
    post_order_vec(root)
    
    # Total log-likelihood is sum over root states and patterns
    # L[root] is (4, K)
    # For each pattern k, the likelihood is sum_s L[root][s, k]
    # Total log-likelihood = sum_k count_k * log(sum_s L[root][s, k])
    
    root_likelihoods = np.sum(L[root], axis=0) # shape (K,)
    # Avoid log(0)
    root_likelihoods = np.maximum(root_likelihoods, 1e-10)
    
    log_ll = np.sum(pattern_counts * np.log(root_likelihoods))
    
    return log_ll

def _optimize_branch_lengths(tree_adj, pattern_matrix, pattern_counts):
    """Optimize branch lengths to maximize log-likelihood."""
    edges = []
    edge_indices = {}
    for u in tree_adj:
        for v in tree_adj[u]:
            if u < v:
                edges.append((u, v))
                edge_indices[(u, v)] = len(edges) - 1
                edge_indices[(v, u)] = len(edges) - 1
                
    n_edges = len(edges)
    if n_edges == 0:
        return {}, 0.0
        
    def neg_log_likelihood(x):
        # x is array of branch lengths
        branch_lengths = {}
        for i, (u, v) in enumerate(edges):
            branch_lengths[(u, v)] = x[i]
            branch_lengths[(v, u)] = x[i]
        ll = _calc_likelihood(tree_adj, branch_lengths, pattern_matrix, pattern_counts)
        return -ll
        
    # Initial guess: small positive values
    x0 = np.full(n_edges, 0.1)
    
    # Bounds: [0, 5] (JC distance max is ~3, but branch lengths can be larger in unrooted? No, max distance is bounded.)
    bounds = [(0.0, 5.0) for _ in range(n_edges)]
    
    try:
        res = minimize(neg_log_likelihood, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-5})
        if res.success:
            best_lengths = {}
            for i, (u, v) in enumerate(edges):
                best_lengths[(u, v)] = res.x[i]
                best_lengths[(v, u)] = res.x[i]
            return best_lengths, -res.fun
        else:
            # Return initial or best found
            best_lengths = {}
            for i, (u, v) in enumerate(edges):
                best_lengths[(u, v)] = x0[i]
                best_lengths[(v, u)] = x0[i]
            ll = _calc_likelihood(tree_adj, best_lengths, pattern_matrix, pattern_counts)
            return best_lengths, ll
    except Exception:
        best_lengths = {}
        for i, (u, v) in enumerate(edges):
            best_lengths[(u, v)] = x0[i]
            best_lengths[(v, u)] = x0[i]
        ll = _calc_likelihood(tree_adj, best_lengths, pattern_matrix, pattern_counts)
        return best_lengths, ll

def _nj_tree(labels, distances):
    """Run Neighbor Joining to get initial tree structure (adjacency list)."""
    n = len(labels)
    if n <= 3:
        # Star tree
        adj = {i: {} for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i != j:
                    adj[i][j] = 0.1
        return adj
        
    nodes = list(range(n))
    # distances is dict {(i,j): d}
    # We need to maintain a distance matrix for the current set of nodes
    # Let's use a 2D array for speed
    # Map node names to indices in the current list
    # Initially, nodes are 0..n-1
    
    # dist[i][j]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = distances.get((i,j), distances.get((j,i), 0))
                
    adj = {i: {} for i in range(n)}
    
    while len(nodes) > 3:
        m = len(nodes)
        # Calculate Q matrix
        # r[i] = sum_j dist[i][j]
        r = np.sum(dist, axis=1)
        
        best_q = float('inf')
        best_i = -1
        best_j = -1
        
        for i_idx in range(m):
            i = nodes[i_idx]
            for j_idx in range(i_idx + 1, m):
                j = nodes[j_idx]
                q = (m - 2) * dist[i][j] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i = i
                    best_j = j
                    
        # Join i and j into new node n
        new_node = n
        n += 1
        nodes = [x for x in nodes if x not in (best_i, best_j)]
        nodes.append(new_node)
        
        # Calculate branch lengths
        # len(i, new) = 0.5 * dist[i][j] + 1/(2(m-2)) * (r[i] - r[j])
        # len(j, new) = 0.5 * dist[i][j] - 1/(2(m-2)) * (r[i] - r[j])
        len_i_new = 0.5 * dist[best_i][best_j] + (r[best_i] - r[best_j]) / (2 * (m - 2))
        len_j_new = 0.5 * dist[best_i][best_j] - (r[best_i] - r[best_j]) / (2 * (m - 2))
        
        # Ensure positive
        len_i_new = max(len_i_new, 0.01)
        len_j_new = max(len_j_new, 0.01)
        
        # Update adjacency
        adj[best_i][new_node] = len_i_new
        adj[new_node][best_i] = len_i_new
        adj[best_j][new_node] = len_j_new
        adj[new_node][best_j] = len_j_new
        
        # Update distances for new node
        # dist[new][k] = 0.5 * (dist[i][k] + dist[j][k] - dist[i][j])
        # We need to expand dist matrix? Or just use dict for new distances.
        # Since we are in a loop, let's just use a dict for the current active nodes' distances.
        # But we already have a 2D array. Let's just create a new one.
        
        new_nodes_list = nodes
        new_m = len(new_nodes_list)
        new_dist = np.zeros((new_m, new_m))
        node_to_idx = {node: idx for idx, node in enumerate(new_nodes_list)}
        
        for idx_i, i in enumerate(new_nodes_list):
            for idx_j, j in enumerate(new_nodes_list):
                if i == j:
                    continue
                if i == new_node:
                    # dist[new][k]
                    k = nodes[idx_j] # Wait, nodes was updated.
                    # We need the old indices for i and j.
                    # This is getting messy.
                    
        # Let's simplify: Use a dict for distances between active nodes.
        # This is slower but easier to manage.
        pass

    # The above NJ implementation is getting complex with array resizing.
    # Let's use a simpler dict-based NJ.
    
    # Restart NJ with dict
    active_nodes = list(range(len(labels)))
    dist_dict = {}
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                dist_dict[(i, j)] = distances.get((i, j), distances.get((j, i), 0))
                
    adj = {i: {} for i in range(len(labels))}
    next_node_id = len(labels)
    
    while len(active_nodes) > 3:
        m = len(active_nodes)
        r = {u: sum(dist_dict.get((u, v), 0) for v in active_nodes if v != u) for u in active_nodes}
        
        best_q = float('inf')
        best_i = -1
        best_j = -1
        
        for i in active_nodes:
            for j in active_nodes:
                if i == j:
                    continue
                q = (m - 2) * dist_dict.get((i, j), 0) - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i = i
                    best_j = j
                    
        # Join best_i and best_j
        new_node = next_node_id
        next_node_id += 1
        
        len_i_new = 0.5 * dist_dict.get((best_i, best_j), 0) + (r[best_i] - r[best_j]) / (2 * (m - 2))
        len_j_new = 0.5 * dist_dict.get((best_i, best_j), 0) - (r[best_i] - r[best_j]) / (2 * (m - 2))
        
        len_i_new = max(len_i_new, 0.01)
        len_j_new = max(len_j_new, 0.01)
        
        adj[best_i][new_node] = len_i_new
        adj[new_node][best_i] = len_i_new
        adj[best_j][new_node] = len_j_new
        adj[new_node][best_j] = len_j_new
        
        # Update distances
        for k in active_nodes:
            if k not in (best_i, best_j):
                new_dist_ik = 0.5 * (dist_dict.get((best_i, k), 0) + dist_dict.get((best_j, k), 0) - dist_dict.get((best_i, best_j), 0))
                dist_dict[(new_node, k)] = new_dist_ik
                dist_dict[(k, new_node)] = new_dist_ik
                
        # Remove old nodes, add new
        active_nodes = [x for x in active_nodes if x not in (best_i, best_j)]
        active_nodes.append(new_node)
        
    # Final 3 nodes: a, b, c
    a, b, c = active_nodes
    # Connect them in a star or triangle?
    # In NJ, the last 3 nodes form a triangle.
    # We can just connect them all to each other.
    # But we need a tree, not a graph.
    # The standard NJ stops at 3 nodes and adds the remaining edges.
    # The lengths are determined by the distances.
    # For simplicity, we can just connect a-b, b-c, a-c with lengths derived from distances?
    # No, a tree on 3 nodes has 2 edges.
    # Let's pick a central node, say a.
    # Connect a-b and a-c.
    # Length a-b = 0.5 * (d_ab + d_ac - d_bc)
    # Length a-c = 0.5 * (d_ac + d_ab - d_bc)
    
    d_ab = dist_dict.get((a, b), 0)
    d_ac = dist_dict.get((a, c), 0)
    d_bc = dist_dict.get((b, c), 0)
    
    len_ab = 0.5 * (d_ab + d_ac - d_bc)
    len_ac = 0.5 * (d_ac + d_ab - d_bc)
    
    len_ab = max(len_ab, 0.01)
    len_ac = max(len_ac, 0.01)
    
    adj[a][b] = len_ab
    adj[b][a] = len_ab
    adj[a][c] = len_ac
    adj[c][a] = len_ac
    
    return adj

def _tree_to_newick(adj, labels):
    """Convert adjacency list to Newick string."""
    # Root the tree at 0
    root = 0
    parent = {root: -1}
    children = {i: [] for i in adj}
    
    stack = [root]
    visited = {root}
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                parent[v] = u
                children[u].append(v)
                stack.append(v)
                
    def build_newick(u):
        if not children[u]:
            # Leaf
            return labels[u]
        else:
            child_strs = [build_newick(c) for c in children[u]]
            return "(" + ",".join(child_strs) + ")"
            
    return build_newick(root) + ";"

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 12.0 # Leave 1s margin per instance? Total 270s / 20 = 13.5s.
    
    labels, pattern_matrix, pattern_counts = _build_site_patterns(sequences)
    n_taxa = len(labels)
    
    if n_taxa <= 1:
        return "t0;"
    if n_taxa == 2:
        return "(t0,t1);"
        
    # Compute distances
    distances = {}
    for i in range(n_taxa):
        for j in range(i+1, n_taxa):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            distances[(i, j)] = d
            distances[(j, i)] = d
            
    # Initial NJ Tree
    adj = _nj_tree(labels, distances)
    
    # Optimize branch lengths for initial tree
    branch_lengths, ll = _optimize_branch_lengths(adj, pattern_matrix, pattern_counts)
    
    # NNI Search
    # We need to identify internal edges.
    # An edge is internal if both endpoints have degree > 1?
    # In a tree, leaves have degree 1. Internal nodes have degree >= 3.
    # So an edge is internal if both endpoints are internal nodes.
    
    def get_internal_edges(adj):
        internal_edges = []
        for u in adj:
            if len(adj[u]) > 1: # Internal
                for v in adj[u]:
                    if u < v and len(adj[v]) > 1:
                        internal_edges.append((u, v))
        return internal_edges
        
    best_adj = adj
    best_ll = ll
    best_branch_lengths = branch_lengths
    
    # NNI Loop
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
            
        internal_edges = get_internal_edges(best_adj)
        
        for u, v in internal_edges:
            if time.time() - start_time > time_limit:
                break
                
            # Get neighbors
            neighbors_u = [x for x in best_adj[u] if x != v]
            neighbors_v = [x for x in best_adj[v] if x != v]
            
            if len(neighbors_u) < 2 or len(neighbors_v) < 2:
                continue
                
            # Try swapping neighbors_u[0] with neighbors_v[0]
            # And neighbors_u[1] with neighbors_v[0]?
            # Standard NNI:
            # Current: u-v, u-a, u-b, v-c, v-d
            # Neighbor 1: u-v, u-a, v-b, v-c, u-d (Swap b and c? No, swap u-b and v-c)
            # Let's just generate the 2 alternative topologies.
            
            a, b = neighbors_u
            c, d = neighbors_v
            
            # Alternative 1: Swap b and c
            # u connects to v, a, c
            # v connects to u, b, d
            new_adj_1 = {k: dict(v) for k, v in best_adj.items()}
            # Remove old edges
            del new_adj_1[u][b]
            del new_adj_1[b][u]
            del new_adj_1[v][c]
            del new_adj_1[c][v]
            # Add new edges
            new_adj_1[u][c] = best_adj[u][b] # Use old lengths as init
            new_adj_1[c][u] = best_adj[u][b]
            new_adj_1[v][b] = best_adj[v][c]
            new_adj_1[b][v] = best_adj[v][c]
            
            bl_1, ll_1 = _optimize_branch_lengths(new_adj_1, pattern_matrix, pattern_counts)
            if ll_1 > best_ll:
                best_adj = new_adj_1
                best_ll = ll_1
                best_branch_lengths = bl_1
                improved = True
                break # Restart sweep
                
            # Alternative 2: Swap b and d
            # u connects to v, a, d
            # v connects to u, b, c
            new_adj_2 = {k: dict(v) for k, v in best_adj.items()}
            del new_adj_2[u][b]
            del new_adj_2[b][u]
            del new_adj_2[v][d]
            del new_adj_2[d][v]
            new_adj_2[u][d] = best_adj[u][b]
            new_adj_2[d][u] = best_adj[u][b]
            new_adj_2[v][b] = best_adj[v][d]
            new_adj_2[b][v] = best_adj[v][d]
            
            bl_2, ll_2 = _optimize_branch_lengths(new_adj_2, pattern_matrix, pattern_counts)
            if ll_2 > best_ll:
                best_adj = new_adj_2
                best_ll = ll_2
                best_branch_lengths = bl_2
                improved = True
                break
                
    return _tree_to_newick(best_adj, labels)
