import math

def construct():
    # Generate candidate vertices: numbers in [1, 255] with gcd(x, 255) > 1
    # 255 = 3 * 5 * 17
    candidates = []
    for i in range(1, 256):
        if math.gcd(i, 255) > 1:
            candidates.append(i)
    
    n = len(candidates)
    # Map value to index for faster lookup? Or just use sets of values.
    # Using sets of values is fine for n=127.
    
    # Precompute neighbors for each candidate
    # neighbors[v] is a set of u in candidates such that gcd(u, v) > 1
    neighbors = {}
    for v in candidates:
        neighbors[v] = set()
        for u in candidates:
            if u != v and math.gcd(u, v) > 1:
                neighbors[v].add(u)
                
    # Calculate degrees
    degrees = {v: len(neighbors[v]) for v in candidates}
    
    # Sort candidates by degree descending
    sorted_candidates = sorted(candidates, key=lambda x: degrees[x], reverse=True)
    
    best_clique = set()
    
    # Try starting from the top K candidates
    K = min(50, n)
    for start in sorted_candidates[:K]:
        clique = {start}
        # Candidates that are compatible with start
        current_candidates = neighbors[start].copy()
        
        while current_candidates:
            # Find vertex with highest degree within current_candidates
            # Degree in current_candidates is len(neighbors[v] & current_candidates)
            best_v = None
            max_deg = -1
            for v in current_candidates:
                deg = len(neighbors[v] & current_candidates)
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v is None:
                break
                
            clique.add(best_v)
            # Update current_candidates to only include neighbors of best_v
            current_candidates = current_candidates & neighbors[best_v]
            
        if len(clique) > len(best_clique):
            best_clique = clique
            
    # Ensure 255 is in the result? 
    # The problem states the set must contain 255.
    # Our candidates all have gcd > 1 with 255, so they are compatible with 255.
    # However, the greedy might not pick 255 if it starts elsewhere.
    # We should force 255 into the clique or ensure the search includes it.
    # Actually, the problem asks for a subset containing 255.
    # If the max clique found doesn't contain 255, we can't just add it if it's not connected to all others.
    # But all candidates are connected to 255.
    # So if we find a max clique C in G[V], then C U {255} is a clique in G[V] IF 255 is connected to all in C.
    # Since 255 is in V and connected to all in V, C U {255} is a valid clique.
    # But wait, if 255 is already in C, we are fine.
    # If 255 is not in C, we can add it?
    # The clique C is a subset of V. 255 is in V.
    # If we add 255 to C, is it still a clique?
    # Yes, because 255 is adjacent to all nodes in V.
    # So the max clique in G[V] is either the max clique not containing 255, or one containing 255.
    # Since 255 is universal in G[V] (connected to all other vertices in V? No, only if gcd(u, 255)>1 which is true for all u in V).
    # Wait, is 255 connected to ALL u in V?
    # V is defined as {v | gcd(v, 255) > 1}.
    # So yes, gcd(255, v) > 1 for all v in V.
    # Therefore, 255 is a universal vertex in G[V].
    # The maximum clique in G[V] must contain 255?
    # Not necessarily. A clique not containing 255 could be larger than any clique containing 255?
    # No. If C is a clique not containing 255, then C U {255} is also a clique.
    # So any max clique can be extended to include 255.
    # Thus, the max clique size is 1 + max clique size in G[V \ {255}]?
    # Actually, since 255 is universal, the max clique is just the max clique of the whole graph, and it will always contain 255 in some optimal representation, or we can just add it.
    # So we just need to find the max clique in G[V].
    
    # The loop above finds max cliques.
    # If 255 is not in best_clique, we can add it.
    if 255 not in best_clique:
        best_clique.add(255)
        
    return list(best_clique)
