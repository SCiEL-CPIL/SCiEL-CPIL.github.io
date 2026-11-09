"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assigns a priority score to each vertex such that a greedy decoder
    (picking vertices in descending score order, skipping neighbours of
    already picked vertices) produces a large independent set.
    
    Strategy:
    1. Use a time-limited local search to find a large independent set S.
    2. The priority of a vertex is based on whether it is in S (high score)
       or not (low score). Specifically, vertices in S get a high base score,
       and we can fine-tune the order within S and outside S to maximize
       the chance that the greedy decoder recovers S or a superset.
    
    Actually, a simpler and more robust approach:
    - Run a greedy independent set algorithm (e.g., min-degree) to get a set S.
    - Try to improve S via local search (swap vertices) if time permits.
    - The priority vector is constructed so that vertices in S are picked first.
      Specifically, score[v] = 1.0 if v in S, else 0.0. But this might not be enough
      if the greedy decoder's tie-breaking or order matters.
      
    Better: The greedy decoder picks vertices in descending score order.
    If we assign high scores to vertices in S, and low scores to others,
    the decoder will pick vertices in S first. But if two vertices in S are
    not adjacent, they will both be picked. If they are adjacent, one will be
    picked and the other skipped. But S is an independent set, so no two vertices
    in S are adjacent. So the decoder will pick all vertices in S, and then
    possibly some vertices outside S if they are not adjacent to any picked vertex.
    
    Wait, the decoder skips neighbours of picked vertices. So if we pick all of S,
    then any vertex outside S that is adjacent to any vertex in S is skipped.
    Vertices outside S that are NOT adjacent to any vertex in S can be picked.
    This could only increase the independent set size, which is good.
    
    So the key is to find a large independent set S. The priority function should
    output scores such that the greedy decoder picks S first.
    
    To ensure S is picked first, we can assign:
    score[v] = 1000 - i  for the i-th vertex in S (in some order)
    score[v] = 0          for vertices not in S
    
    But the order within S doesn't matter for the decoder's correctness (since S is independent),
    but it might affect which vertices outside S are skipped. To maximize the chance
    of picking extra vertices outside S, we might want to pick vertices in S that
    have high degree (so they block more outside vertices) or low degree (so they block
    fewer). This is complex.
    
    Simpler: just assign a constant high score to all vertices in S, and a lower score to others.
    The decoder will pick all of S (in arbitrary order), then pick any outside vertex
    not adjacent to S.
    
    Let's implement:
    1. Find a large independent set S using a greedy min-degree algorithm.
    2. Optionally, try to improve S via local search (e.g., swap a vertex in S with
       two vertices outside S if that increases the set size).
    3. Return scores: 1.0 for v in S, 0.0 for v not in S.
    
    However, the greedy decoder picks in descending score order. If all vertices in S have
    the same score, the order is arbitrary (depends on implementation). To be safe, we can
    assign slightly different scores to vertices in S to control the order. But since S is
    independent, the order doesn't affect whether S is fully picked. It only affects which
    outside vertices are blocked.
    
    Let's start with the simple version and see if it works.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([])
    
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Convert adjacency to a set of neighbors for each vertex for faster lookup
    # But n=200, so we can use the matrix directly.
    
    # Step 1: Greedy min-degree independent set
    # We maintain a set of available vertices.
    # At each step, pick the vertex with the minimum degree among available vertices.
    # Remove it and its neighbors from the available set.
    
    available = set(range(n))
    independent_set = []
    
    while available:
        # Find the vertex with minimum degree in the available subgraph
        min_deg = float('inf')
        min_vertex = -1
        for v in available:
            deg = sum(1 for u in available if adjacency[v, u])
            if deg < min_deg:
                min_deg = deg
                min_vertex = v
                if deg == 0:
                    break  # Can't do better than 0
        if min_vertex == -1:
            break
        independent_set.append(min_vertex)
        # Remove min_vertex and its neighbors from available
        neighbors = set(np.where(adjacency[min_vertex])[0])
        available -= neighbors
        available.discard(min_vertex)
        
        if time.time() > deadline:
            break
    
    # Step 2: Local search improvement
    # Try to swap a vertex in S with a vertex outside S if it leads to a larger set.
    # This is complex, so let's try a simpler improvement:
    # After picking S, try to add vertices outside S that are not adjacent to any vertex in S.
    # But the greedy decoder will do this automatically.
    
    # Instead, let's try to improve S by replacing a vertex in S with two vertices outside S.
    # This is a 1-2 swap.
    
    # For simplicity, let's just use the greedy set for now.
    
    # Step 3: Construct the priority vector
    scores = np.zeros(n, dtype=float)
    for v in independent_set:
        scores[v] = 1.0
    
    # To break ties and potentially improve the decoder's behavior, we can
    # assign slightly different scores based on the order in which vertices were picked.
    # But since S is independent, the order doesn't matter for picking S.
    # Let's assign scores in reverse order of picking, so that vertices picked later
    # (which might be more "central") have higher scores.
    
    # Actually, let's just use the simple version first.
    
    return scores
# EVOLVE-BLOCK-END
