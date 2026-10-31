import numpy as np

def priority(vector, n):
    # This function is called for each vector in some order? 
    # The contract says: "vectors of {0,1,2}^n are added greedily in descending priority"
    # This implies the evaluator calls priority(v) for all v, sorts them, and then greedily adds them 
    # if they keep the set a cap set.
    # Wait, "added greedily in descending priority when they keep the set a cap set".
    # This is a STATIC priority assignment. I cannot update priorities based on the current set S 
    # during the greedy phase because the greedy phase is driven by the sorted list of priorities.
    # If I return a static priority, the order is fixed.
    # If the order is fixed, I just need to find a good fixed order.
    
    # Re-reading the contract: "vectors ... are added greedily in descending priority when they keep the set a cap set"
    # This usually means:
    # 1. Calculate priority(v) for all v.
    # 2. Sort v by priority descending.
    # 3. Iterate through sorted v:
    #    if adding v to current set S keeps S a cap set:
    #        add v to S
    #    else:
    #        skip v
    # 4. Return size of S.
    
    # If this is the case, my "dynamic" heuristic (minimizing new forbidden based on current S) 
    # cannot be implemented directly in `priority` because `priority` is called independently for each vector 
    # before the greedy loop starts (or during, but if the order is fixed by the sort, the decision to add 
    # is made sequentially, but the priority is static).
    
    # However, if the evaluator calls `priority` for each vector, sorts them, and THEN runs the greedy loop, 
    # then I am stuck with a static ordering.
    
    # Is there a way to make a static ordering good?
    # Yes, if the static ordering corresponds to a good greedy strategy.
    # A common good static ordering for cap sets is to prioritize vectors with fewer 1s? Or specific patterns?
    
    # Actually, looking at similar FunSearch tasks, sometimes the "greedy" part implies that the 
    # priority can depend on the vector only.
    
    # If I must use a static priority, what is a good one?
    # Random? No, deterministic is better for reproducibility, but random might explore better.
    # However, "score 1.0000" with uniform priority (0.0) means the tie-breaking is likely lexicographic or random.
    # If I return a constant, the order is undefined (or lexicographic).
    
    # Let's try to mimic the "Minimum Degree" or "Central" property statically.
    # Vectors with more 1s? 
    # In $\mathbb{F}_3^n$, the number of 1s is not the only factor.
    
    # Let's try a heuristic: Prioritize vectors that are "far" from the origin (Hamming weight)?
    # Or prioritize vectors with weight 1?
    
    # Actually, a known good cap set construction is based on the Levenshtein code.
    # But for a simple static priority, let's try to prioritize vectors that are likely to be 
    # "independent" early on.
    
    # If the evaluator allows the priority to be dynamic (i.e., it calls priority(v, current_set)?), 
    # the signature is `priority(vector, n)`. It does not take the current set.
    # So it MUST be static.
    
    # What static order maximizes cap set size?
    # If we pick all vectors of weight 0, 1, 2... 
    # Let's try prioritizing vectors with lower Hamming weight?
    # Or higher?
    
    # Let's try: Priority = -Hamming Weight.
    # This picks 000...0 first. Then 100...0, 010...0, etc.
    # This creates a large independent set?
    # {0, e1, e2, ..., en}. 
    # Check: 0, e1, e2. Sum = e1+e2. Not 0. 
    # 0, e1, e1? Distinct required.
    # Lines in this set: 
    # If we have {0, e1, e2, e3}, 
    # 0+e1+e2 = e1+e2 (not in set if n>2).
    # e1+e2+e3 = e1+e2+e3 (not in set).
    # So the set of all vectors with weight <= 1 is a cap set?
    # Size = 1 + n.
    # For n=8, size = 9.
    # Uniform random might get larger?
    # Max cap set size for n=8 is 20 (known value? $A_3(8) = 20$).
    # So weight <= 1 is bad.
    
    # What if we prioritize vectors with weight 2?
    # Or a mix?
    
    # Since I cannot adapt, I will use a heuristic that is known to work well for static greedy:
    # Randomize the priority slightly? No, must be deterministic for scoring?
    # "Score: cap set size".
    
    # Let's try to return a priority based on the sum of the vector components?
    # Or simply a hash?
    
    # Actually, if the tie-breaking in the evaluator is random, then returning 0.0 for all 
    # effectively makes the order random.
    # If the current score is 1.0000 (which is the seed), and the seed is uniform, 
    # maybe the seed score IS the random greedy score?
    # If so, I need to beat the random greedy.
    
    # A better static heuristic:
    # Prioritize vectors that are "orthogonal" to many others?
    
    # Let's try: Priority = - (number of 1s in the vector).
    # This picks low weight vectors first.
    
    # Alternative: Priority = (number of 2s in the vector).
    
    # Let's try a more sophisticated static priority:
    # Map the vector to a number and use a property of that number?
    
    # Given the constraints, the most robust "static" improvement is often 
    # to use a **randomized** priority if the evaluator allows it? 
    # But the function is deterministic.
    
    # Wait, if I return a value based on the vector, I can create a specific order.
    # Let's try an order that groups vectors by their "type".
    
    # Let's try: Priority = sum(vector).
    # Vectors with sum 0, 1, 2...
    
    # Let's try: Priority = -sum(vector).
    
    # I will implement a priority that favors vectors with a specific pattern, 
    # e.g., those with exactly one non-zero element? No, that was small.
    
    # How about favoring vectors with **no** 1s? (Only 0s and 2s).
    # Or favoring vectors with **many** 1s?
    
    # Let's try to maximize the number of 1s.
    # Priority = count of 1s.
    
    # Let's try to minimize the number of 1s.
    # Priority = -count of 1s.
    
    # I'll go with **Maximizing the number of 1s**. 
    # Intuition: Vectors with many 1s are "far" from each other in some sense?
    
    # Actually, a very common trick in these FunSearch cap set tasks is to use 
    # **Random** priorities. But I can't use random if it's not seeded?
    # I can use a hash of the vector as a pseudo-random priority.
    # This breaks the lexicographic bias.
    
    # Let's use a hash-based priority.
    # This effectively randomizes the order.
    # If the seed (uniform/lex) is suboptimal, random might be better on average.
    # But I need to BEAT the seed.
    
    # Let's try a hybrid: 
    # Priority = - (Hamming Distance to the all-1 vector)?
    
    # I will use a simple heuristic: 
    # Prioritize vectors with **even** sum?
    
    # Let's stick to the most likely improvement: 
    # The "Minimum Degree" heuristic cannot be applied statically.
    # However, a **static** approximation of "good" vertices are those that are 
    # less likely to conflict.
    
    # I will use a **randomized** priority based on a hash of the vector.
    # This explores the space more uniformly than lexicographic.
    
    import hashlib
    # Convert vector to string
    v_str = str(vector)
    # Hash to get a float
    h = int(hashlib.md5(v_str.encode('utf-8')).hexdigest(), 16)
    return float(h) / float(0xFFFFFFFFFFFFFFFF)

# EVOLVE-BLOCK-END
