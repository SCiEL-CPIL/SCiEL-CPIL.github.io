def construct():
    n = 5
    # Transitive tournament: i -> j for i < j
    # This ensures that for any set of 4 vertices, the smallest index vertex in the set
    # (or the global source 0 if 0 is in the set, or the smallest in the set) dominates all others in the set?
    # Wait. If S = {1, 2, 3, 4}, vertex 0 is not in S.
    # Does 0 dominate S? Yes, 0 -> 1, 0 -> 2, 0 -> 3, 0 -> 4.
    # So 0 is a common dominator for S.
    # If S = {0, 1, 2, 3}, vertex 4 is not in S.
    # Does 4 dominate S? No, 0 -> 4.
    # Does any vertex dominate S?
    # 1 dominates 2,3. Not 0.
    # 2 dominates 3. Not 0,1.
    # 3 dominates nothing.
    # 0 dominates 1,2,3.
    # So 0 is a common dominator for S={0,1,2,3}?
    # "vertex with arcs to all four".
    # If v=0, arcs to 1,2,3. Does it have arc to 0? No.
    # So v must be OUTSIDE the set.
    # For S={0,1,2,3}, is there a v outside S (v=4) that dominates S?
    # 4 is dominated by 0,1,2,3. So 4 does NOT dominate S.
    # So the transitive tournament on 5 vertices does NOT satisfy "Every 4-set has a common dominator outside the set".
    
    # So my previous deduction was wrong.
    # In a transitive tournament, the source 0 dominates everyone.
    # But if 0 is IN the 4-set, 0 cannot dominate itself.
    # And the next vertex 1 does not dominate 0.
    # So if 0 is in S, no vertex outside S dominates S?
    # Vertices outside S are {4}. 4 is dominated by 0. So 4 does not dominate S.
    # So the transitive tournament FAILS the property "Every 4-set has a common dominator outside the set".
    
    # What if the dominator can be INSIDE the set?
    # "vertex with arcs to all four".
    # If v is in S, it has arcs to 3 vertices. It cannot have an arc to itself.
    # So it cannot have arcs to "all four".
    # So the dominator MUST be outside the set.
    
    # Therefore, the property is: For all S subset V, |S|=4, exists v in V\S such that v->u for all u in S.
    
    # Let's check n=5 again.
    # Is there ANY tournament on 5 vertices satisfying this?
    # If v is a dominator for S, v must have out-degree at least 4 in the whole graph? No, just to S.
    # But if v dominates S, and S has size 4, v must have out-degree at least 4 (if v is not in S).
    # So every vertex that is not in S must dominate S.
    # This implies that for any S, the complement V\S must contain a vertex that dominates S.
    # This is a very strong condition.
    
    # Let's check the seed: Paley 83.
    # Paley tournaments are highly symmetric.
    # It is known that Paley tournaments have the property that the neighborhood of any vertex is a tournament with certain properties.
    
    # If I can't find a small n easily, I should stick to the seed or try a slightly smaller prime?
    # Primes p=3, 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83.
    # The seed is 83.
    # Maybe 67 is the "smallest known" mentioned in the prompt?
    # "The smallest known has 67 vertices".
    # This suggests that a tournament on 67 vertices is a valid instance.
    # And probably smaller ones (like 5, 7, 11) are NOT valid.
    # Why? Because for small n, it's hard to satisfy "Every 4-set has a common dominator outside".
    # As n grows, it becomes easier?
    # Actually, for large n, random tournaments might fail?
    # No, if the property is "Every 4-set has a common dominator", it's a "covering" condition.
    
    # If the prompt says "smallest known is 67", I should try to output 67.
    # Or maybe 31 (the lower bound)?
    # If 31 is the lower bound, it means f(4) >= 31.
    # If I find a valid tournament on 31, I prove f(4) <= 31.
    # If f(4) is the threshold for "every tournament has the property", then a single valid tournament does NOT prove f(4) <= 31.
    # But if the task is "Find the smallest n such that there EXISTS a tournament on n vertices with the property", then 31 is a candidate.
    
    # Given the seed is 83, and "smallest known is 67", I will try 67.
    # 67 is a prime.
    # I will use the Paley tournament on 67 vertices.
    
    p = 67
    squares = {x * x % p for x in range(1, p)}
    return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}
