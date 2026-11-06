"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
        S_set = set(S)
        # Optimization: check pairs. If a+b=c, then c-a=b.
        # Since a,b,c are distinct in the set, we can just iterate.
        # For small sets (<=240), O(n^2) is fine.
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def get_conflicts(S):
        """Return a set of pairs (a, b) such that a+b is in S. 
        Actually, for local search, we need to know which elements cause conflicts.
        Let's represent the state as a list of elements.
        A conflict is a triple (a, b, c) where a+b=c.
        For efficiency in SA, we can just check validity of the whole set.
        Given n<=240, is_sum_free is O(n^2) ~ 57k ops. Very fast.
        """
        pass

    def solve_one(values, time_limit):
        """
        Find a large sum-free subset of values.
        Strategy:
        1. Start with a heuristic subset (e.g., all odds).
        2. Use Simulated Annealing to improve.
        Moves: 
           - Add an element if it doesn't create a conflict.
           - Remove an element if it resolves conflicts (or just to allow swap).
           - Swap an element in with an element out.
        
        Actually, a simpler and effective approach for sum-free sets:
        The set of all odd numbers is sum-free.
        The set of numbers in (n/3, n] is sum-free (for positive integers).
        But here we have mixed signs.
        
        Let's use a greedy construction with multiple random restarts and local search.
        Local Search Move: 
        - Try to add an element x not in S. If valid, add it.
        - If S is "stuck" (no elements can be added), try swapping an element y in S with x not in S.
          If S - {y} + {x} is sum-free, and maybe larger or same size but "better" (hard to define better), accept.
          Actually, we just want to maximize size. So if size increases, accept.
          If size stays same, accept with probability based on temperature (to escape local optima).
          
        However, simply maximizing size is the goal.
        A known good lower bound is roughly n/2 for random sets? No, n/3 is guaranteed.
        For random sets, the maximum sum-free subset is often close to n/2.
        
        Let's implement a simple hill climbing / SA.
        State: Set S.
        Objective: Maximize |S| subject to sum-free constraint.
        
        Initial state: Greedy.
        1. Sort values by absolute value? Or random?
        2. Try to add elements one by one.
        
        Let's try a different heuristic:
        Partition into 3 sets based on modulo 3?
        Sum-free property: If a ≡ b (mod 3) and a+b ≡ 2a (mod 3), it's not automatically sum-free.
        Odd numbers work because odd+odd=even.
        
        Let's stick to SA on the set of elements.
        """
        n = len(values)
        values_set = set(values)
        
        # Heuristic initialization
        # 1. Odds
        odds = [x for x in values if x % 2 != 0]
        # 2. Positives (if all positive, large half is sum-free)
        #    But we have negatives.
        #    Let's just use odds as baseline.
        
        best_S = set(odds)
        best_size = len(best_S)
        
        # If odds are empty or very small, try evens? No, evens+evens=even.
        # Try numbers with high absolute value?
        
        # Let's do a greedy fill first.
        # Randomize order to get different starts.
        
        rng = random.Random(42)
        
        # Multi-start
        best_overall = set()
        best_size_overall = 0
        
        # Time check
        if time.time() > time_limit:
            return sorted(best_overall)
            
        # Run SA for a fixed number of iterations or until time runs out
        # We have ~110s total for 3 instances. So ~35s each.
        
        current_S = set(odds)
        
        # Greedy improvement of current_S
        # Try to add any element from values_set - current_S
        remaining = values_set - current_S
        added = True
        while added:
            added = False
            # Randomize remaining to break symmetry
            rem_list = list(remaining)
            rng.shuffle(rem_list)
            for x in rem_list:
                if x in current_S: continue
                # Check if adding x makes it sum-free
                # x+a=b or x+b=a or a+x=b ... essentially x+y=z
                # We need to check if there exist a,b in current_S such that a+b=x or x+a=b or x+b=a.
                # This is equivalent to checking if x is in the sumset of current_S, 
                # or if x + current_S intersects current_S.
                # i.e., (x + current_S) & current_S is not empty?
                # If a+b=x, then b = x-a. So x-a in current_S.
                # If a+x=b, then b-x=a. So b-x in current_S.
                # If x+b=a, then a-x=b. So a-x in current_S.
                
                # Efficient check:
                # For each a in current_S, check if x-a in current_S (covers a+b=x)
                # For each a in current_S, check if a-x in current_S (covers x+b=a => b=a-x)
                # Note: a+b=x and x+b=a are symmetric if we just check differences.
                # If x is in S+S, conflict.
                # If x-S intersects S, conflict.
                
                conflict = False
                for a in current_S:
                    if (x - a) in current_S:
                        conflict = True
                        break
                if not conflict:
                    current_S.add(x)
                    remaining.remove(x)
                    added = True
            
        # Now try local search (swaps) to improve size?
        # Actually, if we could add any element, we did. So current_S is "maximal" w.r.t additions.
        # But it might not be maximum. We might need to remove some elements to add more.
        
        # SA Loop
        # State: current_S
        # Move: Swap an element in S with an element not in S.
        # Accept if new size > old size.
        # If new size == old size, accept with prob exp(-delta/T)? 
        # Since size is the only objective, delta is 0 for same size swaps.
        # We want to explore same-size swaps to find a configuration where we can add more.
        
        # Let's define a "cost" function that is 0 if sum-free, infinity otherwise.
        # We only move to valid states.
        
        # To find a better state:
        # 1. Pick random y in current_S.
        # 2. Pick random x in values_set - current_S.
        # 3. Try S' = S - {y} + {x}.
        # 4. If S' is sum-free, accept.
        #    - If |S'| > |S|, always accept.
        #    - If |S'| == |S|, accept with probability p (to wander).
        
        # Since we are stuck at a local maximum size, we need to wander in the landscape of same-size sum-free sets.
        
        T = 1.0
        T_min = 0.01
        alpha = 0.99
        
        iter_count = 0
        max_iters = 50000 # Adjust based on time
        
        while time.time() < time_limit:
            if iter_count % 1000 == 0:
                T *= alpha
                if T < T_min:
                    T = T_min
            
            # Pick random swap
            if not current_S:
                break
                
            y = rng.choice(list(current_S))
            # Pick x from remaining
            rem = values_set - current_S
            if not rem:
                break
            x = rng.choice(list(rem))
            
            if x == y:
                continue
                
            # Try swap
            # Remove y, add x
            # Check if S' is sum-free
            # S' = (S - {y}) U {x}
            
            # Optimization: 
            # Removing y can only remove conflicts.
            # Adding x can only add conflicts.
            # Conflicts involving x:
            # 1. x + a = b (a,b in S-y) => b-x = a in S-y
            # 2. a + b = x (a,b in S-y) => x-a = b in S-y
            
            # We can check validity of S' directly.
            # Given n is small, is_sum_free is fast.
            
            S_test = set(current_S)
            S_test.remove(y)
            S_test.add(x)
            
            if is_sum_free(S_test):
                # Accept?
                # Size is same.
                # We accept to explore.
                # To avoid cycling, we can use a simple random walk.
                current_S = S_test
                iter_count += 1
            else:
                iter_count += 1
                
            # Check time
            if time.time() > time_limit:
                break
            
            # Track best
            if len(current_S) > best_size_overall:
                best_size_overall = len(current_S)
                best_overall = set(current_S)
                
        return sorted(best_overall)

    instances = [
        instance(60, 801),
        instance(120, 802),
        instance(240, 803)
    ]
    
    results = []
    
    # Allocate time roughly equally, but give more to larger instances if needed?
    # 60 is small, 240 is large.
    # Let's just split the remaining time.
    
    # Simpler: Just run solve_one for each with a sub-deadline.
    # But we need to ensure we finish all 3.
    
    # Let's give 30s to each, but check global deadline.
    
    for i, vals in enumerate(instances):
        # Calculate remaining time
        rem_time = deadline - time.time()
        if rem_time < 1:
            # Not enough time, return what we have (empty or heuristic)
            results.append([x for x in vals if x % 2 != 0])
            continue
            
        # Allocate time for this instance
        # If first instance, give it more? No, just equal.
        # But if we run out of time in the middle, we should return the best found so far.
        
        # Let's run solve_one with a limit of rem_time / (3 - i) - 1
        sub_deadline = time.time() + max(1.0, rem_time / (3 - i) - 0.5)
        
        res = solve_one(vals, sub_deadline)
        results.append(res)
        
    return results
# EVOLVE-BLOCK-END
