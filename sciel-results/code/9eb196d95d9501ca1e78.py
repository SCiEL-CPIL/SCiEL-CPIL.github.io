import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0 # 5s safety margin
    
    n = 12
    m = 40
    all_masks = [1 << i for i in range(n)]
    
    # Helper: check if a is subset of b
    def is_subset(a, b):
        return (a & b) == a

    # Helper: check if comparable
    def is_comparable(a, b):
        if a == b: return False # distinct sets
        return is_subset(a, b) or is_subset(b, a)

    def count_comparable(sets):
        count = 0
        n_sets = len(sets)
        for i in range(n_sets):
            for j in range(i + 1, n_sets):
                if is_comparable(sets[i], sets[j]):
                    count += 1
        return count

    # Initialize
    # Main chain
    chain = []
    mask = 0
    for i in range(n + 1):
        chain.append(mask)
        if i < n:
            mask |= (1 << i)
    
    # Add random sets to reach 40
    current_sets = set(chain)
    while len(current_sets) < m:
        # Bias towards small sets or chain neighbors?
        # Just random for now
        rand_mask = random.randint(0, (1 << n) - 1)
        if rand_mask not in current_sets:
            current_sets.add(rand_mask)
            
    current_sets = list(current_sets)
    
    # Calculate initial score and degrees
    def calc_degrees(sets):
        degs = [0] * len(sets)
        for i in range(len(sets)):
            for j in range(len(sets)):
                if i != j and is_comparable(sets[i], sets[j]):
                    degs[i] += 1
        return degs

    degs = calc_degrees(current_sets)
    best_score = sum(degs) // 2
    
    # Local Search
    while time.time() < deadline:
        i = random.randint(0, m - 1)
        old_set = current_sets[i]
        
        # Try to find a better replacement
        # We can try a few random candidates
        improved = False
        for _ in range(10): # Try 10 candidates per step
            new_set = random.randint(0, (1 << n) - 1)
            if new_set in current_sets:
                continue
                
            # Calculate gain
            gain = 0
            for j in range(m):
                if j != i:
                    if is_comparable(new_set, current_sets[j]):
                        gain += 1
            
            # Loss is degs[i]
            if gain > degs[i]:
                # Accept
                current_sets[i] = new_set
                # Update degrees
                # This is O(N^2) if we recalc all, but we can do O(N)
                # degs[i] becomes gain
                # For all j != i:
                #   if new_set ~ current_sets[j]: degs[j] += 1
                #   if old_set ~ current_sets[j]: degs[j] -= 1
                #   else: no change?
                # Wait, degs[j] counts comparability with ALL others.
                # So we must update degs[j] for all j != i.
                
                for j in range(m):
                    if j == i:
                        continue
                    old_comp = is_comparable(old_set, current_sets[j])
                    new_comp = is_comparable(new_set, current_sets[j])
                    if old_comp and not new_comp:
                        degs[j] -= 1
                    elif new_comp and not old_comp:
                        degs[j] += 1
                    # if both or neither, no change
                
                degs[i] = gain
                best_score += (gain - degs[i] + degs[i]) # Wait, best_score update
                # Old contribution of i was degs[i] (old). New is gain.
                # But degs[i] in the array was the old value.
                # We updated degs[j] for j!=i.
                # The total score is sum(degs)//2.
                # Delta score = (Gain - OldDeg)
                # Because pairs (i, j) change.
                # Pairs (j, k) for j,k != i do not change.
                # So total score changes by Gain - OldDeg.
                best_score += gain - degs[i] # Error: degs[i] was updated?
                # Let's just recalc score if needed, or track carefully.
                # Actually, simpler:
                # score = sum(degs) // 2
                # Let's just keep a variable 'current_score'
                
                improved = True
                break
        
        # If no improvement, we might get stuck.
        # Add a restart or random jump?
        # For now, just continue.
        
    # Convert to lists
    result = []
    for mask in current_sets:
        result.append([b for b in range(n) if (mask >> b) & 1])
        
    return result
