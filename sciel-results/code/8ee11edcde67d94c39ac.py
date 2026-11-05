import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 140  # Safety margin

    def gen_instance(size, seed, limit=10000):
        rng = random.Random(seed)
        vals = set()
        while len(vals) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                vals.add(x)
        return sorted(vals)

    def count_triples_involving(S, x):
        """
        Count how many valid sum triples (a+b=c) exist in S that involve x.
        If x is in S, these are the triples that will be removed if x is removed.
        If x is not in S, these are the triples that would be created if x is added.
        """
        count = 0
        # Case 1: x is the sum (c). a + b = x
        # We need to count pairs (a, b) in S such that a + b = x.
        # Since a, b are in S, and distinctness of a,b is not required for the count logic 
        # (we just count the number of such pairs), we iterate.
        # Optimization: If S is a set, we can check (x-a) in S.
        for a in S:
            b = x - a
            if b in S:
                # We found a pair (a, b).
                # Does order matter? The triple is {a, b, x}.
                # The condition is "no a+b=c".
                # If a!=b, (a,b) and (b,a) are the same triple.
                # If a==b, it's just one triple (2a=x).
                # So we should count each unordered pair {a,b} once.
                # To avoid double counting a!=b:
                if a < b:
                    count += 1
                elif a == b:
                    count += 1
        
        # Case 2: x is an addend. x + b = c or a + x = c.
        # This is equivalent to: exists b, c in S such that x + b = c.
        # i.e., c - x = b.
        # So for each c in S, check if (c - x) in S.
        for c in S:
            b = c - x
            if b in S:
                # Triple is (x, b, c).
                # x is fixed. b and c are in S.
                # Is it possible that x==b or x==c?
                # If x==b, then 2x=c. Triple (x, x, c).
                # If x==c, then x+b=x => b=0. But 0 not in S.
                # So x!=c always.
                # If x!=b, the pair (x,b) is distinct from (b,x)?
                # The triple is defined by the set {x, b, c} and the equation.
                # Since x is fixed as one of the addends, and b is the other,
                # and c is the sum, this is a unique configuration for this x.
                # Do we double count?
                # The triple is (x, b, c).
                # In Case 1, we counted triples where x is the SUM.
                # Here x is an ADDEND.
                # So no overlap with Case 1.
                # Within Case 2, for a fixed x, each valid c gives a unique b.
                # So we just count the number of such c.
                count += 1
        
        return count

    def solve_instance(elements, seed_off, budget):
        t_start = time.time()
        V = list(elements)
        V_set = set(V)
        n = len(V)
        
        # Pre-compute nothing? The count_triples_involving is O(|S|).
        # Total cost per step: N * |S|.
        
        best_S = set()
        best_score = -10**9
        
        # Initializations
        inits = []
        # 1. All odds
        odds = [x for x in V if x % 2 != 0]
        inits.append(set(odds))
        # 2. Greedy by value desc
        S_temp = set()
        for x in sorted(V, reverse=True):
            if count_triples_involving(S_temp, x) == 0:
                S_temp.add(x)
        inits.append(S_temp)
        # 3. Greedy by value asc
        S_temp = set()
        for x in sorted(V):
            if count_triples_involving(S_temp, x) == 0:
                S_temp.add(x)
        inits.append(S_temp)
        # 4. Random greedy
        rng = random.Random(seed_off)
        for _ in range(5):
            S_temp = set()
            rand_V = V[:]
            rng.shuffle(rand_V)
            for x in rand_V:
                if count_triples_involving(S_temp, x) == 0:
                    S_temp.add(x)
            inits.append(S_temp)
            
        # WalkSAT
        for init in inits:
            S = set(init)
            # Ensure S is sum-free? The greedy above ensures it.
            # If not, clean it up.
            # For safety, let's verify and clean if necessary.
            # But greedy with count_triples_involving==0 ensures no violations.
            
            current_score = len(S) # Violations are 0
            
            if current_score > best_score:
                best_score = current_score
                best_S = set(S)
                
            # Local Search
            steps = 0
            while time.time() - t_start < budget and steps < 5000:
                # If we have time, try to improve
                # Calculate delta score for each x in V
                best_delta = -10**9
                best_x = None
                # We prefer moves that increase size or reduce violations.
                # Score = Size - 1000 * Violations
                
                # If current S has violations (shouldn't happen if we start valid and only make valid moves, 
                # but if we do random moves, it might)
                # Let's assume we maintain validity if possible.
                
                # Find best move
                for x in V:
                    if x in S:
                        # Remove x
                        # Delta Size = -1
                        # Delta Violations = - (count of triples in S involving x)
                        # Since S is valid, count is 0.
                        # So Delta Violations = 0.
                        # Delta Score = -1.
                        d_score = -1
                    else:
                        # Add x
                        # Delta Size = +1
                        # Delta Violations = + (count of triples created by adding x to S)
                        # If count is 0, Delta Violations = 0. Delta Score = 1.
                        # If count > 0, Delta Violations > 0. Delta Score = 1 - 1000*count.
                        c = count_triples_involving(S, x)
                        if c == 0:
                            d_score = 1
                        else:
                            d_score = 1 - 1000 * c
                    
                    if d_score > best_delta:
                        best_delta = d_score
                        best_x = x
                
                # If best_delta is 1, we found an addition that keeps validity. Take it.
                # If best_delta is -1, we found a removal (or no addition possible).
                # If best_delta is very negative, we are forced to make a bad move?
                
                if best_x is None:
                    break
                
                # Apply move
                if best_x in S:
                    S.discard(best_x)
                else:
                    S.add(best_x)
                
                # After move, if we added and created violations, we need to handle them.
                # But our heuristic avoids adding if it creates violations (d_score would be negative).
                # So if we added, it was valid.
                # If we removed, it was valid (removal always keeps validity).
                
                # If we are stuck (best_delta <= 0), we might need to escape.
                # The logic above:
                # If valid and can add -> d=1.
                # If valid and cannot add -> all additions d<0, all removals d=-1.
                # So best_delta will be -1 (removal).
                # This causes the set to shrink.
                # Then, after shrinking, we might be able to add different elements.
                # This is the "Remove one, try to add two" logic implicitly.
                
                # To speed up escape, if best_delta == -1 (removal), 
                # we can pick a random element to remove instead of the first one found,
                # or we can just let it pick one.
                # Let's add a small probability to pick a random move if stuck.
                if best_delta <= 0:
                    # We are removing an element.
                    # To diversify, if we have been removing for a while, 
                    # maybe pick randomly?
                    # For now, let's just follow the greedy delta.
                    pass
                
                steps += 1
                
                # Update best
                # Check if S is valid? It should be.
                # But let's be safe.
                # Actually, if we only add when c=0, and remove (which is safe), S is always valid.
                cur_size = len(S)
                if cur_size > best_score:
                    best_score = cur_size
                    best_S = set(S)
                    
                # If we reached a maximal valid set (cannot add, best_delta=-1),
                # and we have time, we can try a "kick":
                # Remove 2 random elements, then greedily add.
                # This is expensive. Let's stick to single-step WalkSAT.
                
        return list(best_S)

    instances = [
        gen_instance(60, 801),
        gen_instance(120, 802),
        gen_instance(240, 803),
    ]

    results = []
    budgets = [10, 30, 90]
    
    for i, elems in enumerate(instances):
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed - 5
        if remaining < 5:
            # Fallback
            res = [x for x in elems if x % 2 != 0]
            results.append(res)
            continue
            
        b = min(budgets[i], max(5, remaining))
        res = solve_instance(elems, seed_off=801 + i * 7, budget=b)
        
        # Verify
        S_set = set(res)
        valid = True
        L = list(res)
        for a in L:
            for b in L:
                if (a + b) in S_set:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            res = [x for x in elems if x % 2 != 0]
        
        results.append(res)
        
    return results
