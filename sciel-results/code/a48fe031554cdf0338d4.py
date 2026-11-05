import time
import random
import itertools
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 18
    all_vertices = (1 << n) - 1
    
    # Precompute bit masks for fast operations
    # adj_mask[i] is a bitmask of neighbors of i in H
    # We want to minimize the largest S such that H[S] is triangle-free AND alpha(H[S]) <= 2
    # This is equivalent to finding the largest S where G[S] (complement) is triangle-free AND alpha(G[S]) <= 2?
    # Wait, let's clarify the definitions.
    # G is K4-free.
    # Score is max |S| such that G[S] is triangle-free.
    # G[S] is triangle-free <=> H[S] has no independent set of size 3 (alpha(H[S]) <= 2).
    # So we want to minimize the max |S| such that alpha(H[S]) <= 2.
    # Note: The constraint "G is K4-free" means H has no independent set of size 4.
    # So we are looking for the largest subset S of V such that alpha(H[S]) <= 2,
    # subject to the global constraint alpha(H) <= 3.
    
    # We will represent H by a list of bitmasks adj_mask.
    
    def is_valid_h(adj_mask):
        # Check if alpha(H) <= 3.
        # i.e., no independent set of size 4.
        # We can check this by iterating all 4-subsets. C(18,4) = 3060. This is fast enough for a few checks.
        # But for local search, we need to be faster.
        # For now, let's use a fast check.
        # An independent set of size 4 in H means 4 vertices with no edges between them in H.
        # In G, this is a K4.
        # So we just need to ensure G has no K4.
        # Let's stick to checking alpha(H) <= 3.
        for quad in itertools.combinations(range(n), 4):
            u, v, w, x = quad
            # Check if no edges in H among these 4
            if not (adj_mask[u] & (1 << v) or adj_mask[u] & (1 << w) or adj_mask[u] & (1 << x) or
                    adj_mask[v] & (1 << w) or adj_mask[v] & (1 << x) or
                    adj_mask[w] & (1 << x)):
                return False
        return True

    def calc_score_fast(adj_mask):
        # Find max size of S such that alpha(H[S]) <= 2.
        # This is equivalent to finding the largest induced subgraph of H with independence number <= 2.
        # We can use a branch and bound or a greedy approximation for speed, 
        # but the task says "exact branch and bound" is used by the evaluator.
        # To be safe and fast, let's implement a reasonably fast exact check for sizes 5, 6, etc.
        # If we find a subset of size 5 with alpha <= 2, score is at least 5.
        # We want to minimize this.
        # So we check from high to low.
        # Max possible is 18.
        # But since alpha(H) <= 3, the score is at most 3? 
        # No. alpha(H) <= 3 means there is NO independent set of size 4 in the WHOLE graph.
        # So for any S, alpha(H[S]) <= alpha(H) <= 3.
        # Wait. If alpha(H) <= 3, then the max S such that alpha(H[S]) <= 2 could be large.
        # Example: H is a matching. alpha(H) is large. 
        # If H is such that every 4-set has an edge, then alpha(H) <= 3.
        # Does this imply that there is a large S with alpha(H[S]) <= 2?
        # Yes, it's possible.
        # We need to find the max |S| such that H[S] has no independent set of size 3.
        
        # Let's check sizes from 10 down to 0? No, that's too slow.
        # Let's check 5, 6, 7... 
        # Actually, if we can achieve score 4, that's great.
        # Let's check if there exists a subset of size 5 with alpha <= 2.
        # If yes, score >= 5.
        # If no, check size 4.
        
        # To speed this up, we can use a simple greedy or heuristic for the "does there exist S of size k with alpha <= 2" question.
        # But we need the exact score for the evaluator.
        # The evaluator will run its own exact B&B.
        # My job is to construct G. The score is determined by the evaluator.
        # So I don't need to calculate the score exactly myself, I just need to minimize it.
        # However, to guide the search, I need a proxy.
        # A good proxy is the maximum size of an induced subgraph with alpha <= 2.
        # Let's use a fast heuristic to estimate this.
        
        # Heuristic: 
        # Start with all vertices.
        # If there is an independent set of size 3 in H[S], remove one vertex from it?
        # No, that's not how it works.
        
        # Let's just use a simple exact check for small k (e.g., up to 6) and assume if we can't find one, it's smaller.
        # Actually, let's just check for the existence of a "good" subset of size 5.
        # If none exists, the score is <= 4.
        # If one exists, the score is >= 5.
        
        # Let's check size 5 first.
        for subset in itertools.combinations(range(n), 5):
            # Check if alpha(H[subset]) <= 2
            # i.e., no 3 vertices in subset are independent in H.
            ok = True
            for triple in itertools.combinations(subset, 3):
                u, v, w = triple
                if not (adj_mask[u] & (1 << v) or adj_mask[u] & (1 << w) or adj_mask[v] & (1 << w)):
                    ok = False
                    break
            if ok:
                return 5 # At least 5
        
        # Check size 4
        for subset in itertools.combinations(range(n), 4):
            ok = True
            for triple in itertools.combinations(subset, 3):
                u, v, w = triple
                if not (adj_mask[u] & (1 << v) or adj_mask[u] & (1 << w) or adj_mask[v] & (1 << w)):
                    ok = False
                    break
            if ok:
                return 4 # At least 4
                
        return 3 # Or lower, but likely 3 or 4 is the floor for valid graphs

    def get_edges_from_adj_mask(adj_mask):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj_mask[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # Strategy:
    # 1. Generate a random regular graph H of degree d.
    # 2. Check if alpha(H) <= 3. If not, repair or discard.
    # 3. Use local search to minimize the score (proxy: max subset size with alpha <= 2).
    
    best_score = 18
    best_adj_mask = None
    best_edges = []
    
    # Try different degrees
    for d in [8, 9, 10, 11]:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
                
            # Generate random d-regular graph
            adj_mask = [0] * n
            # Simple method: shuffle edges and pick
            # Or use a configuration model approximation
            # Let's just use random edges and then adjust degrees? 
            # Or simpler: random graph with p = d/(n-1)
            p = d / (n - 1)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
                        
            # Ensure it's a valid H (alpha <= 3)
            # If not, try to fix it by adding edges to H (removing from G)
            # Adding edges to H reduces alpha(H).
            # So if alpha(H) > 3, we need to add edges to H.
            
            # Check validity
            if not is_valid_h(adj_mask):
                # Try to fix by adding random edges to H
                fixed = False
                for _ in range(100):
                    if is_valid_h(adj_mask):
                        fixed = True
                        break
                    # Pick a random non-edge in H (edge in G) and add it to H
                    i, j = random.sample(range(n), 2)
                    if not (adj_mask[i] & (1 << j)):
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
                
                if not fixed:
                    continue
                    
            # Local Search
            current_score = calc_score_fast(adj_mask)
            current_adj_mask = [m for m in adj_mask]
            
            for _ in range(200):
                if time.time() > deadline - 2:
                    break
                    
                # Try 1-flips: remove an edge from H (add to G) or add an edge to H (remove from G)
                # We want to minimize score.
                # Removing edges from H (making H sparser) increases alpha(H[S]) potentially, so score might increase.
                # Adding edges to H (making H denser) decreases alpha(H[S]), so score might decrease.
                # But we must maintain alpha(H) <= 3.
                
                # So we should primarily try ADDING edges to H.
                i, j = random.sample(range(n), 2)
                if not (current_adj_mask[i] & (1 << j)):
                    # Try adding edge to H
                    new_adj_mask = [m for m in current_adj_mask]
                    new_adj_mask[i] |= (1 << j)
                    new_adj_mask[j] |= (1 << i)
                    
                    if is_valid_h(new_adj_mask):
                        new_score = calc_score_fast(new_adj_mask)
                        if new_score < current_score:
                            current_score = new_score
                            current_adj_mask = new_adj_mask
                            if current_score < best_score:
                                best_score = current_score
                                best_adj_mask = [m for m in current_adj_mask]
                                if best_score == 4:
                                    best_edges = get_edges_from_adj_mask([0]*n) # Wait, G is complement of H
                                    # G edges are where H has no edge
                                    g_edges = []
                                    for u in range(n):
                                        for v in range(u+1, n):
                                            if not (best_adj_mask[u] & (1 << v)):
                                                g_edges.append([u, v])
                                    return g_edges
                else:
                    # Try removing edge from H (if it doesn't violate alpha(H)<=3? 
                    # Removing edge from H increases alpha(H), so it might violate alpha(H)<=3.
                    # So we should be careful.
                    pass
            
            # Update best if improved
            if current_score < best_score:
                best_score = current_score
                best_adj_mask = [m for m in current_adj_mask]
                
    # Convert best H to G edges
    if best_adj_mask is None:
        # Fallback: random graph
        adj_mask = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        # Force validity
        while not is_valid_h(adj_mask):
            i, j = random.sample(range(n), 2)
            if not (adj_mask[i] & (1 << j)):
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
        best_adj_mask = adj_mask

    g_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (best_adj_mask[u] & (1 << v)):
                g_edges.append([u, v])
                
    return g_edges
