import time
import random

def construct():
    n = 12
    start_time = time.time()
    deadline = start_time + 85
    
    best_a = [1 << i for i in range(n)]
    best_max = max(best_a)
    
    def get_forbidden(a_prefix):
        m = len(a_prefix)
        if m == 0:
            return set([0]), set([0])
        
        sums = set([0])
        for x in a_prefix:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums.update(new_sums)
        
        diffs = set()
        sums_list = list(sums)
        for s1 in sums_list:
            for s2 in sums_list:
                diffs.add(s1 - s2)
        return diffs, sums

    def greedy_complete(a_prefix, perturbations):
        """
        perturbations: dict {step_index: amount_to_add}
        """
        a = a_prefix[:]
        for k in range(len(a), n):
            forbidden, _ = get_forbidden(a)
            val = 1
            while True:
                if val not in forbidden and val not in a:
                    break
                val += 1
            
            if k in perturbations:
                val += perturbations[k]
                while val in forbidden or val in a:
                    val += 1
                    
            a.append(val)
        return a

    # Initial greedy run (no perturbations)
    # This should give powers of 2
    # a_0 = 1. forbidden={0,1,-1}. val=2.
    # a_1 = 2. forbidden={-3..3}. val=4.
    # ...
    
    # Search loop
    while time.time() < deadline:
        # Generate random perturbations
        # Try 1 to 3 perturbations
        num_pert = random.randint(1, 3)
        pert_steps = random.sample(range(1, n), num_pert)
        perturbations = {}
        for step in pert_steps:
            # Small random perturbation
            # Positive or negative?
            # Negative might make val <= 0 or duplicate.
            # Let's try positive and small negative.
            # If negative, we need to ensure val > 0 and not in a.
            delta = random.randint(-5, 10)
            perturbations[step] = delta
            
        candidate = greedy_complete([], perturbations)
        
        # Check validity just in case (should be valid by construction)
        # But if delta was negative, we might have skipped a valid smaller number?
        # No, we start from the greedy val and add delta.
        # If delta is negative, we might land on a forbidden number, and the while loop fixes it by incrementing.
        # So it's valid.
        
        max_val = max(candidate)
        if max_val < best_max:
            best_max = max_val
            best_a = candidate
            
    return best_a
