"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign a score to each vertex for greedy MIS selection.
    
    Strategy: Model vertex priority as expected survival in a random sequential
    elimination process. A vertex is 'safe' (likely to be picked) if it has few
    neighbors that are themselves likely to be picked early. We approximate this
    by iterating a belief propagation-like update where the score of a vertex
    depends on the scores of its neighbors (specifically, the probability that
    at least one neighbor is picked before it).
    
    For the greedy decoder (descending score, skip neighbors of picked),
    a vertex with high score is picked first. If it's picked, its neighbors are
    skipped. We want to pick vertices that 'waste' the fewest other high-potential
    vertices. This is similar to minimizing the sum of neighbor scores, but
    dynamic.
    
    We use a simple iterative refinement:
    1. Start with base priority: 1 / (1 + degree). Low degree is good.
    2. Update: score[i] = 1 / (1 + sum(score[j] for j in neighbors(i)) * w)
       This penalizes vertices whose neighbors are also high-scoring (dense clusters).
       Actually, if neighbors are high-scoring, they might get picked first, blocking me.
       Or, if I am high-scoring, I block them.
       
    Let's think about the objective: Maximize number of picked vertices.
    Greedy picks highest score.
    If A and B are connected, and score(A) > score(B), A is picked, B is skipped.
    To maximize count, we should prefer picking 'leaf-like' nodes or nodes in sparse areas.
    
    A known good heuristic for MIS is the 'smallest degree first' (which is what the seed does: -degree).
    However, local structures matter. A node connected to two high-degree nodes is 'worse' than a node
    connected to two low-degree nodes?
    If I pick a node, I remove its neighbors.
    The 'cost' of picking a node is the number of neighbors it removes.
    But those neighbors might have been picked anyway if they were high priority.
    
    Let's try a score based on the 'independent set potential' of the neighborhood.
    
    Algorithm:
    1. Compute degrees.
    2. Initialize scores s_i = 1.0
    3. Iterate a few times:
       For each i, compute neighbor_score_sum = sum(s_j for j in N(i))
       The 'pressure' on i is related to how likely its neighbors are to be selected.
       If neighbors are selected, i is blocked.
       If i is selected, neighbors are blocked.
       
       A common approach in MaxSAT or similar is to use a weighted degree.
       
    Let's try: score_i = 1 / (1 + degree_i * (1 + avg_neighbor_degree))
    This penalizes nodes in dense regions more heavily.
    
    Alternative:
    Use the eigenvalues of the adjacency matrix?
    The Fiedler vector (2nd smallest eigenvalue) often identifies community structure.
    Vertices with small components in the Fiedler vector are often 'central' or in dense clusters.
    We want to pick vertices that are 'peripheral'.
    
    Let's test the 'peripheral' hypothesis.
    Compute the eigenvector corresponding to the smallest eigenvalue (or second smallest if connected).
    Actually, for sparse graphs, the smallest eigenvalue is close to 0.
    The components of the eigenvector for the smallest eigenvalue indicate the sign structure.
    
    Let's stick to a simpler, robust heuristic that captures 'density of neighborhood'.
    
    score_i = 1 / (1 + degree_i + beta * sum(degrees of neighbors_i))
    
    This is a 2-hop degree. It penalizes nodes that are in dense areas (high 2-hop degree).
    
    Let's refine:
    base = degree
    two_hop = adjacency @ degree  (sum of degrees of neighbors)
    score = 1 / (1 + base + 0.5 * two_hop)
    
    This should prioritize nodes that are isolated from dense clusters.
    
    We also need to handle the case where the graph is disconnected.
    
    Let's add a small random tie-breaker? No, deterministic is better for reproducibility in evaluation if possible, but the task says 'mean independent-set size on five generated... graphs', implying fixed graphs or a distribution. The evaluator runs the function on specific graphs.
    
    Let's try the 2-hop degree heuristic.
    """
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    # Degree: number of neighbors
    degree = adj.sum(axis=1)
    
    # 2-hop degree: sum of degrees of neighbors
    # This measures how 'dense' the neighborhood is.
    two_hop = adj @ degree
    
    # Combine:
    # We want low degree and low 2-hop degree to have high score.
    # Score = 1 / (1 + degree + alpha * two_hop)
    
    # Normalize two_hop to be comparable to degree?
    # Max two_hop can be much larger.
    # Let's scale it.
    max_two_hop = two_hop.max() if n > 0 else 1.0
    if max_two_hop > 0:
        two_hop_norm = two_hop / max_two_hop
    else:
        two_hop_norm = two_hop
        
    # Tune alpha.
    # If alpha is too high, we over-penalize dense areas.
    # If alpha is too low, it's just degree.
    
    # Let's try a weighted sum in the denominator.
    # denom = 1 + degree + 0.1 * two_hop_norm
    
    # Actually, let's look at the 'expected survivors' idea.
    # If we pick a random vertex v, the expected number of vertices we lose is its degree.
    # But if we pick a vertex with high degree neighbors, those neighbors are 'expensive' to lose?
    # No, we lose them regardless.
    # The issue is that picking a high-degree vertex removes many vertices that *could* have formed a large independent set among themselves?
    # No, independent set vertices don't conflict with each other.
    # The conflict is only with neighbors.
    
    # Let's consider the 'greedy' nature.
    # If I pick v, I remove N(v).
    # The size of the remaining graph is n - 1 - |N(v)|.
    # The potential MIS size of the remaining graph is roughly (n - 1 - |N(v)|) / (avg_degree_remaining + 1).
    # This is complex to compute iteratively.
    
    # Let's go with the 2-hop degree as a proxy for 'structural importance' or 'centrality'.
    # Peripheral nodes (low degree, low 2-hop degree) are safer to pick because they block fewer 'important' nodes?
    # Or rather, they are less likely to be blocked by others?
    
    # Let's try: score = - (degree + 0.5 * two_hop)
    # This is a smooth extension of degree.
    
    # Let's also consider the 'clustering coefficient' or local density.
    # But 2-hop is cheap.
    
    # Let's try a few variations and pick one.
    # Variation 1: -degree (Seed)
    # Variation 2: -(degree + two_hop)
    # Variation 3: 1 / (1 + degree + two_hop)
    
    # Let's use Variation 3, normalized.
    
    # To avoid division by zero or large numbers, we normalize.
    # denom = degree + two_hop_norm
    # score = 1 / (1 + denom)
    
    # Let's try to be more sophisticated:
    # Use the inverse of the (degree + 1) as the base, and adjust by neighborhood density.
    
    # Actually, a very strong heuristic for MIS is the 'Largest First' or 'Smallest First' with tie-breaking by '2-hop degree'.
    # Since we output a single score, we can combine them.
    
    # score = - (degree * 100 + two_hop_norm)
    # This prioritizes low degree first, then low 2-hop.
    
    # Let's try:
    # score = - (degree + 0.1 * two_hop_norm)
    
    # Wait, the seed is -degree.
    # If we just add a small penalty for 2-hop, it might improve ties.
    
    # Let's try a more aggressive penalty for dense areas.
    # score = - (degree + 1.0 * two_hop_norm)
    
    # Let's calculate:
    # degree: int
    # two_hop_norm: [0, 1]
    
    # If degree is 10, and two_hop_norm is 1, score = -11.
    # If degree is 10, and two_hop_norm is 0, score = -10.
    # So it prefers the one with lower 2-hop.
    
    # Let's try this.
    
    # Also, we should handle the case where n=0.
    if n == 0:
        return np.array([])
        
    # Compute 2-hop
    two_hop = adj @ degree
    
    # Normalize 2-hop to [0, 1]
    max_2h = two_hop.max()
    if max_2h > 0:
        two_hop_norm = two_hop / max_2h
    else:
        two_hop_norm = np.zeros_like(two_hop, dtype=float)
        
    # Combine
    # Weight for 2-hop
    w_2h = 0.5
    
    score = -(degree.astype(float) + w_2h * two_hop_norm)
    
    return score

# EVOLVE-BLOCK-END
