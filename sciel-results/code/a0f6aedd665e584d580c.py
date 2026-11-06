import random

def construct():
    N = 180
    
    # Pre-compute all reciprocal triples {a, b, c} with a < b < c
    triples = []
    for a in range(1, N + 1):
        max_b = min(N, 2 * a - 1)
        for b in range(a + 1, max_b + 1):
            denom = b - a
            if (a * b) % denom == 0:
                c = (a * b) // denom
                if c > b and c <= N:
                    triples.append((a, b, c))
    
    # Build adjacency: for each element, store the other two elements in each triple it belongs to
    # neighbors[e] = list of (u, v) such that {e, u, v} is a triple
    neighbors = {i: [] for i in range(1, N + 1)}
    for a, b, c in triples:
        neighbors[a].append((b, c))
        neighbors[b].append((a, c))
        neighbors[c].append((a, b))
        
    best_size = 0
    best_set = set()
    
    # Number of restarts. N=180 is small, so we can do many.
    # Time limit is 60s. Each run is very fast.
    # Let's try 200 restarts.
    num_restarts = 200
    
    for _ in range(num_restarts):
        # Shuffle elements
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        
        S = set()
        # conflict_count[e] = number of triples involving e where the other two are in S
        conflict_count = {i: 0 for i in range(1, N + 1)}
        
        for e in elements:
            if conflict_count[e] == 0:
                # Safe to add e
                S.add(e)
                # Update conflict counts for other elements
                for u, v in neighbors[e]:
                    # The triple is {e, u, v}
                    # If u is in S, then v's conflict count increases (because e is now in S)
                    # If v is in S, then u's conflict count increases
                    if u in S:
                        conflict_count[v] += 1
                    if v in S:
                        conflict_count[u] += 1
                    # Note: u and v cannot both be in S at this moment if they formed a triple with e?
                    # No, they could be in S if they didn't form a triple with each other and e wasn't in S yet.
                    # But if u and v are both in S, then {u, v, e} is a triple.
                    # If u and v are both in S, then when we added u, v's count might have increased?
                    # Let's trace:
                    # We add e.
                    # For a neighbor pair (u, v):
                    #   If u in S: The pair (e, v) is now "active" for v? No.
                    #   The triple is {e, u, v}.
                    #   We are adding e.
                    #   For element v: if u is already in S, then the triple {e, u, v} becomes "blocked" for v?
                    #   Actually, we need to know if adding e creates a conflict for v.
                    #   A conflict for v exists if there is a triple {v, x, y} where x, y in S.
                    #   The triple {e, u, v} involves v, e, u.
                    #   Since e is being added, if u is in S, then v now has a conflict (e, u).
                    #   So conflict_count[v] should increase by 1.
                    #   Similarly, if v is in S, u now has a conflict (e, v).
                    #   What if both u and v are in S?
                    #   Then v gets +1 (due to e, u) and u gets +1 (due to e, v).
                    #   Is this correct?
                    #   Yes, because the triple {e, u, v} now has two elements in S (e and u) for v, and (e and v) for u.
                    #   Wait, if u and v are both in S, then {u, v} was already a pair in S.
                    #   Did we check for conflict between u and v when adding them?
                    #   When we added u, v was not in S yet (or was it?).
                    #   If v was in S when u was added, then u would have had a conflict count > 0 and would NOT have been added.
                    #   So u and v cannot both be in S if they form a triple with some other element?
                    #   No, they can be in S if they don't form a triple with each other.
                    #   But here they DO form a triple with e.
                    #   So if u and v are both in S, it means they were added earlier.
                    #   When u was added, v was in S. Did u have a conflict?
                    #   The triple is {e, u, v}. For u to have a conflict at addition time, e would need to be in S.
                    #   But e is being added NOW. So e was not in S.
                    #   So u could have been added safely even if v was in S, because e was not in S.
                    #   So it is possible that both u and v are in S.
                    #   In that case, adding e creates a conflict for BOTH u and v.
                    #   So we increment both.
                    
                    # However, we must ensure we don't double count or miss updates.
                    # The logic holds:
                    # If u in S: v is now in conflict with (e, u).
                    # If v in S: u is now in conflict with (e, v).
                    
                    # One edge case: What if u == v? No, elements are distinct in triples.
                    
                    # Optimization: We only update if the count was 0? No, we need accurate counts.
                    # But if conflict_count[x] > 0, we won't add x anyway.
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
    return list(best_set)
