"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    # Safety margin: return before 90s hard limit. 
    # We use 85s to be safe, but typically this finishes in <5s.
    deadline = start_time + 85.0
    
    n = 180
    
    # 1. Precompute all forbidden triples (a, b, c) such that 1/a = 1/b + 1/c
    # and a, b, c are distinct and in [1, 180].
    # Note: The equation 1/a = 1/b + 1/c implies a < b and a < c.
    # So a is the smallest. We can iterate a from 1 to n, and b, c from a+1 to n.
    # Actually, we just need to know which triples are forbidden.
    # For the SAT/Annealing approach, we need to count violations.
    # A violation occurs if a, b, c are all in the set.
    
    # Let's precompute for each pair (b, c) with b < c, the value a = bc/(b+c).
    # If a is an integer, a != b, a != c, and 1 <= a <= n, then (a, b, c) is a forbidden triple.
    # We can store these as a list of tuples (a, b, c).
    
    triples = []
    # To make violation counting fast, we can also build a map:
    # For each a, list of (b, c) pairs that conflict with a.
    # Or for each pair (b, c), the conflicting a.
    
    # Let's just store the list of triples.
    # Number of triples is small (bounded by O(n^2)).
    
    for b in range(1, n + 1):
        for c in range(b + 1, n + 1):
            prod = b * c
            ssum = b + c
            if prod % ssum == 0:
                a = prod // ssum
                if 1 <= a <= n and a != b and a != c:
                    # Ensure a < b and a < c? 
                    # If 1/a = 1/b + 1/c, then a is harmonic mean-ish.
                    # Since b,c >= 1, 1/b + 1/c <= 2. So 1/a <= 2 => a >= 0.5.
                    # Also 1/a < 1/b => a > b. Wait.
                    # 1/a = 1/b + 1/c. Since c > 0, 1/a > 1/b => a < b.
                    # Similarly a < c.
                    # So a is always the smallest.
                    if a < b and a < c:
                        triples.append((a, b, c))
    
    # 2. Simulated Annealing
    # State: A bitmask or list of booleans representing inclusion of 1..180.
    # We use a list of 0s and 1s for simplicity and speed in Python loops?
    # Or a set of active elements?
    # A set is good for O(1) lookup, but iterating over all n elements to count violations is O(n^2) if naive.
    # Better: Maintain a count of active violations.
    
    # Let's define the energy function:
    # E = - (number of elements in set) + lambda * (number of violated triples)
    # We want to maximize size, so we minimize -size + penalty.
    # If we ensure no violations, we just maximize size.
    # In SA, we allow violations if they lead to a larger set, hoping to resolve them later.
    
    # Initial state: Random subset or Greedy valid subset?
    # Starting with a valid subset is safer.
    # Let's generate a valid subset using a simple greedy method first.
    
    def greedy_valid_subset():
        S = set()
        # Try adding elements in random order
        order = list(range(1, n + 1))
        random.shuffle(order)
        for x in order:
            # Check if adding x creates a triple
            # A triple (a, b, c) is created if x is one of a, b, c and the other two are in S.
            # Since triples are precomputed, we can check:
            # For each triple (a, b, c) in triples:
            # If x == a and b in S and c in S -> conflict
            # If x == b and a in S and c in S -> conflict
            # If x == c and a in S and b in S -> conflict
            
            # To speed this up, we can precompute for each x, the list of triples involving x.
            # triples_by_element[x] = list of (other1, other2, role)
            
            # Let's precompute this outside.
            # But for now, let's just check all triples? 
            # Number of triples is ~few thousand. n=180. 
            # Greedy is O(n * #triples) ~ 180 * 2000 = 360k ops. Fast enough.
            
            valid = True
            for a, b, c in triples:
                if x == a:
                    if b in S and c in S:
                        valid = False
                        break
                elif x == b:
                    if a in S and c in S:
                        valid = False
                        break
                elif x == c:
                    if a in S and b in S:
                        valid = False
                        break
            
            if valid:
                S.add(x)
        return S

    # Precompute triples_by_element for faster updates during SA
    triples_by_element = {i: [] for i in range(1, n + 1)}
    for a, b, c in triples:
        # Store (other1, other2)
        # If we toggle x, we need to know which other elements form a triple with x.
        # If x == a, others are b, c.
        # If x == b, others are a, c.
        # If x == c, others are a, b.
        
        triples_by_element[a].append((b, c))
        triples_by_element[b].append((a, c))
        triples_by_element[c].append((a, b))

    def count_violations(S):
        """Count number of triples fully contained in S."""
        count = 0
        for a, b, c in triples:
            if a in S and b in S and c in S:
                count += 1
        return count

    def get_energy(S):
        """Energy = -|S| + penalty * violations. We want to minimize this."""
        # If we use a large penalty, we effectively force validity.
        # But for SA to explore, we might want a moderate penalty.
        # Actually, we want to maximize |S| subject to violations=0.
        # We can just track |S| and violations separately.
        # Acceptance probability:
        # If new_state has 0 violations and larger size: Accept.
        # If new_state has 0 violations and smaller size: Accept with prob exp(-delta_size / T).
        # If new_state has violations: Accept with prob exp(-(delta_violations * P + delta_size) / T).
        
        pass

    # Let's run SA
    best_set = greedy_valid_subset()
    best_size = len(best_set)
    
    current_set = set(best_set)
    current_violations = 0 # Should be 0 from greedy
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-4
    
    # To speed up violation update, we can maintain a count of active violations.
    # However, updating count exactly when toggling one element is tricky 
    # because a triple might have multiple elements toggled? No, we toggle one.
    # When we add x:
    #   New violations = number of triples (x, u, v) where u, v in S.
    # When we remove x:
    #   Removed violations = number of triples (x, u, v) where u, v in S (before removal).
    
    # Let's implement a fast delta calculation.
    
    def delta_violations_add(x, S):
        """Returns number of new violations if x is added to S."""
        new_v = 0
        for u, v in triples_by_element[x]:
            if u in S and v in S:
                new_v += 1
        return new_v

    def delta_violations_remove(x, S):
        """Returns number of violations removed if x is removed from S (x is in S)."""
        # Note: S still contains x here.
        # We check triples involving x. If the other two are in S, it's a violation.
        rem_v = 0
        for u, v in triples_by_element[x]:
            if u in S and v in S:
                rem_v += 1
        return rem_v

    # Main SA Loop
    steps = 0
    max_steps = 50000 # Adjust based on time
    
    while time.time() < deadline - 2.0:
        # Pick a random element to toggle
        x = random.randint(1, n)
        
        if x in current_set:
            # Try to remove
            # Delta Size: -1
            # Delta Violations: - (violations removed)
            v_rem = delta_violations_remove(x, current_set)
            new_violations = current_violations - v_rem
            new_size = len(current_set) - 1
            
            # Calculate acceptance
            # We prefer valid sets.
            # If current is valid (0 viol) and new is valid (0 viol):
            #   Accept if new_size > current_size (impossible, size decreases) OR with prob exp(-1/T) to allow shrinking?
            #   Usually we don't want to shrink if valid. But sometimes shrinking allows growing elsewhere.
            #   Standard SA on size: delta = -1. Accept with prob exp(-1/T).
            
            # If current has violations and new reduces them:
            #   Likely accept.
            
            # Let's define a cost function: Cost = 100 * violations - size.
            # We want to minimize Cost.
            # Delta Cost = 100 * (new_v - curr_v) - (new_size - curr_size)
            # If x removed:
            #   Delta V = -v_rem
            #   Delta S = -1
            #   Delta C = 100 * (-v_rem) - (-1) = 1 - 100 * v_rem
            
            delta_cost = 1 - 100 * v_rem
            
            if delta_cost < 0:
                # Improvement
                accept = True
            else:
                # Worse
                accept = random.random() < math.exp(-delta_cost / T)
            
            if accept:
                current_set.remove(x)
                current_violations = new_violations
                if current_violations == 0 and len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
        
        else:
            # Try to add
            v_add = delta_violations_add(x, current_set)
            new_violations = current_violations + v_add
            new_size = len(current_set) + 1
            
            # Delta Cost = 100 * v_add - 1
            delta_cost = 100 * v_add - 1
            
            if delta_cost < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta_cost / T)
            
            if accept:
                current_set.add(x)
                current_violations = new_violations
                if current_violations == 0 and len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
        
        steps += 1
        T *= cooling_rate
        
        # Restart if stuck?
        # If T is very low, we might get stuck in a local minimum.
        # But since we track best_set, we are safe.
        # We can also do a "kick" by randomizing a few elements.
        if steps % 1000 == 0:
            # Check if we are making progress
            pass

    # Final Verification
    # Ensure best_set is valid
    valid = True
    for a, b, c in triples:
        if a in best_set and b in best_set and c in best_set:
            valid = False
            break
    
    if not valid:
        # Fallback to greedy if SA failed to keep it valid (shouldn't happen if we only update best when valid)
        # But just in case, run greedy again
        best_set = greedy_valid_subset()
    
    return list(best_set)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    print(len(res))
