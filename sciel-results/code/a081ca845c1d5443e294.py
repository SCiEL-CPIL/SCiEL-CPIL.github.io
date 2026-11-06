import random
import math
from itertools import combinations

def construct():
    N_MAX = 192
    TARGET_SIZE = 48
    
    def is_valid_set(S):
        """
        Check if set S is valid.
        S is a list of distinct integers.
        Returns True if no element divides the sum of 3 distinct others.
        """
        S_set = set(S)
        if len(S) < 4:
            return True
            
        for a in S:
            others = [x for x in S if x != a]
            if len(others) < 3:
                continue
                
            # Count residues of 'others' modulo a
            res_count = [0] * a
            for x in others:
                res_count[x % a] += 1
                
            # Check all pairs u, v in others
            # We need to find w in others \ {u, v} such that (u + v + w) % a == 0
            # i.e., w % a == (- (u + v)) % a
            
            # Optimization: Instead of all pairs, we can just check if such a triple exists.
            # But we need to ensure distinctness.
            
            # Iterate u
            for i in range(len(others)):
                u = others[i]
                # Iterate v
                for j in range(i + 1, len(others)):
                    v = others[j]
                    target_res = (-(u + v)) % a
                    # We need a w in others, w != u, w != v, with w % a == target_res
                    count = res_count[target_res]
                    # Subtract u if it matches
                    if u % a == target_res:
                        count -= 1
                    # Subtract v if it matches
                    if v % a == target_res:
                        count -= 1
                    
                    if count > 0:
                        return False
        return True

    def repair(S):
        """
        Remove elements from S until it is valid.
        Prioritizes removing smaller numbers? Or random?
        Let's remove elements involved in conflicts.
        """
        S = list(S)
        valid = False
        while not is_valid_set(S):
            # Find a conflict
            # We need to find a, u, v, w in S such that a | u+v+w and distinct.
            # We can reuse the logic from is_valid_set to find the first conflict.
            conflict_found = False
            for a in S:
                others = [x for x in S if x != a]
                if len(others) < 3:
                    continue
                res_count = [0] * a
                for x in others:
                    res_count[x % a] += 1
                
                for i in range(len(others)):
                    u = others[i]
                    for j in range(i + 1, len(others)):
                        v = others[j]
                        target_res = (-(u + v)) % a
                        count = res_count[target_res]
                        if u % a == target_res:
                            count -= 1
                        if v % a == target_res:
                            count -= 1
                        
                        if count > 0:
                            # Conflict found: a, u, v, and some w
                            # Remove a or one of u, v, w.
                            # Let's remove a (the divisor) as it might be "stronger"
                            # Or remove a random one of {a, u, v}
                            S.remove(a)
                            conflict_found = True
                            break
                    if conflict_found:
                        break
                if conflict_found:
                    break
            
            if not conflict_found:
                # Should not happen if is_valid_set is consistent
                break
                
        return S

    def greedy_construct(random_order=True):
        """
        Construct a valid set using greedy approach.
        """
        S = []
        candidates = list(range(1, N_MAX + 1))
        if random_order:
            random.shuffle(candidates)
        else:
            candidates.sort(reverse=True)
            
        for x in candidates:
            # Check if x can be added to S
            # We need to check if adding x breaks validity for x (as divisor) and for existing elements (as divisors of x+u+v)
            
            # 1. Check x as divisor:
            # Does x divide sum of any 3 distinct elements in S?
            # We can check this using the same residue logic, but S is the "others" for x.
            # Wait, x is new. So "others" for x are S.
            # We need to check if there exist u,v,w in S such that x | u+v+w.
            
            if len(S) >= 3:
                res_count = [0] * x
                for u in S:
                    res_count[u % x] += 1
                
                # Check if there is a triple in S summing to 0 mod x
                # This is equivalent to checking if there are 3 elements in S with residues summing to 0 mod x.
                # We can just iterate pairs in S.
                found = False
                for i in range(len(S)):
                    u = S[i]
                    for j in range(i + 1, len(S)):
                        v = S[j]
                        target = (-(u + v)) % x
                        count = res_count[target]
                        if u % x == target:
                            count -= 1
                        if v % x == target:
                            count -= 1
                        if count > 0:
                            found = True
                            break
                    if found:
                        break
                if found:
                    continue # x cannot be added because it would divide a sum of 3 in S
            
            # 2. Check existing a in S as divisor:
            # Does a divide x + u + v for any u,v in S \ {a}?
            for a in S:
                others = [x] + [y for y in S if y != a]
                # We need to check if there exist u,v in others (distinct, not a) such that a | u+v+w?
                # No, the condition is: a divides sum of 3 DISTINCT OTHERS.
                # The others are from S_new = S + [x].
                # The 3 others must be distinct from a.
                # So we pick 3 from S_new \ {a}.
                # One of them is x.
                # So we need to check if there exist u, v in S \ {a} such that a | x + u + v.
                
                if len(S) < 2:
                    continue
                
                # We need to check if there is a pair u,v in S \ {a} such that (x + u + v) % a == 0.
                # i.e., (u + v) % a == (-x) % a.
                
                res_count = [0] * a
                for y in S:
                    if y != a:
                        res_count[y % a] += 1
                
                target = (-x) % a
                # We need to find u, v in S \ {a} such that u%a + v%a == target (mod a)
                # And u != v.
                
                # Iterate u in S \ {a}
                valid_pair = False
                for u in S:
                    if u == a:
                        continue
                    r_u = u % a
                    r_v_needed = (target - r_u) % a
                    # We need a v in S \ {a} such that v % a == r_v_needed and v != u.
                    count = res_count[r_v_needed]
                    if r_u == r_v_needed:
                        # u and v must be distinct, so we need at least 2 elements with this residue
                        # But we are picking u first, so we need 1 more.
                        # If count includes u, then count-1 is the number of others.
                        # So if count >= 2, there is a v.
                        # Wait, if r_u == r_v_needed, then v must have the same residue.
                        # We need v != u.
                        # So we need at least one other element with residue r_v_needed.
                        if count >= 2:
                            valid_pair = True
                            break
                    else:
                        # u and v have different residues.
                        # We just need at least one element with residue r_v_needed.
                        # It can be u? No, r_u != r_v_needed, so it's a different element.
                        if count >= 1:
                            valid_pair = True
                            break
                            
                if valid_pair:
                    # Conflict: a divides x + u + v
                    break
            else:
                # No conflict with existing a
                S.append(x)
                
        return S

    # GA Parameters
    pop_size = 50
    generations = 100
    
    # Initialize population
    population = []
    for i in range(pop_size):
        # Mix of greedy (sorted) and greedy (random)
        if i % 2 == 0:
            S = greedy_construct(random_order=False)
        else:
            S = greedy_construct(random_order=True)
        population.append(S)
    
    best_S = max(population, key=len)
    
    for gen in range(generations):
        # Sort by size
        population.sort(key=len, reverse=True)
        
        # Selection: Keep top half, replace bottom half
        # Or tournament selection
        
        new_population = population[:pop_size // 2]
        
        while len(new_population) < pop_size:
            # Pick two parents
            p1 = random.choice(population[:pop_size // 2])
            p2 = random.choice(population[:pop_size // 2])
            
            # Crossover: Union and Repair
            # Combine elements
            combined = list(set(p1 + p2))
            
            # Repair
            child = repair(combined)
            
            # Mutation
            if random.random() < 0.5:
                # Add random element
                candidates = [x for x in range(1, N_MAX + 1) if x not in child]
                if candidates:
                    x = random.choice(candidates)
                    # Try to add, if invalid, repair
                    child_new = child + [x]
                    child = repair(child_new)
            
            if random.random() < 0.5:
                # Remove random element
                if child:
                    idx = random.randint(0, len(child) - 1)
                    child = child[:idx] + child[idx+1:]
            
            new_population.append(child)
            
        population = new_population
        current_best = max(population, key=len)
        if len(current_best) > len(best_S):
            best_S = current_best
            
        # Early stop if we reached max possible?
        if len(best_S) == TARGET_SIZE:
            break
            
    return best_S
