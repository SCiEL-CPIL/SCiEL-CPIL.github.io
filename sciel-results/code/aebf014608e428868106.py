import random
import time

def construct():
    n = 128
    target_start = 128
    target_end = 256
    
    # Helper to build counts from scratch (for initialization)
    def build_counts(A_set):
        counts = [0] * (2 * n + 1)
        A_list = list(A_set)
        for i, a in enumerate(A_list):
            for b in A_list[i:]:
                counts[a + b] += 1
        return counts

    # Helper to update counts incrementally
    def add_element(A_set, counts, x):
        # x is not in A_set yet
        for y in A_set:
            counts[x + y] += 1
        counts[2 * x] += 1
        A_set.add(x)

    def remove_element(A_set, counts, x):
        # x is in A_set
        for y in A_set:
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        A_set.discard(x)

    def get_peak(counts):
        return max(counts[target_start:target_end + 1])

    def is_valid(counts):
        for s in range(target_start, target_end + 1):
            if counts[s] == 0:
                return False
        return True

    start_time = time.time()
    timeout = 55.0

    best_A = None
    best_peak = float('inf')

    # Phase 1: Constructive Greedy
    # Start with a small set. 
    # To cover [128, 256], we need sums to reach 128 and 256.
    # 0+128=128, 128+128=256.
    # Let's start with {0, 128}.
    
    A = set()
    counts = [0] * (2 * n + 1)
    
    # Seed with 0 and 128 to anchor the range
    add_element(A, counts, 0)
    add_element(A, counts, 128)
    
    # Current state:
    # 0+0=0, 0+128=128, 128+128=256.
    # Counts: count[0]=1, count[128]=1, count[256]=1.
    # Missing: 129..255.
    
    while True:
        if time.time() - start_time > timeout:
            break
            
        # Find missing sums or sums with low count
        missing = [s for s in range(target_start, target_end + 1) if counts[s] == 0]
        
        if not missing:
            # All covered. Break to local search
            break
        
        # Pick the sum with the lowest count (0 is lowest, but we picked 0s above)
        # If no missing, pick the one with min count > 0 to balance?
        # The prompt says "Every integer ... must be a sum". So we MUST cover 0s first.
        # Once all are > 0, we can try to balance.
        # For now, just cover 0s.
        s_target = missing[0]
        
        # Find best pair (a, b) to add to cover s_target
        # a+b = s_target, a <= b, 0 <= a, b <= 128
        best_pair = None
        best_cost = float('inf')
        
        for a in range(max(0, s_target - 128), min(128, s_target // 2) + 1):
            b = s_target - a
            if b > 128:
                continue
            if a > b:
                continue
            
            # Calculate cost of adding a and b to A
            # Cost = increase in peak + penalty for new elements
            new_A = A.copy()
            new_counts = counts[:]
            
            # Tentatively add a
            if a not in new_A:
                for y in new_A:
                    new_counts[a + y] += 1
                new_counts[2 * a] += 1
                new_A.add(a)
            
            # Tentatively add b
            if b not in new_A:
                for y in new_A:
                    new_counts[b + y] += 1
                new_counts[2 * b] += 1
                new_A.add(b)
            
            new_peak = get_peak(new_counts)
            # Penalty for size to keep A small initially? 
            # Actually, we want to minimize peak. Size is secondary but helps efficiency.
            cost = new_peak * 1000 + len(new_A)
            
            if cost < best_cost:
                best_cost = cost
                best_pair = (a, b)
        
        if best_pair is None:
            break
        
        a, b = best_pair
        if a not in A:
            add_element(A, counts, a)
        if b not in A:
            add_element(A, counts, b)
            
        # Update best if valid
        if is_valid(counts):
            current_peak = get_peak(counts)
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = A.copy()

    # If we didn't finish constructive phase in time, use what we have
    if best_A is None:
        if is_valid(counts):
            best_A = A.copy()
            best_peak = get_peak(counts)
        else:
            # Fallback: use the seed from original program logic if valid
            A_fallback = set(range(64, 129))
            counts_fallback = build_counts(A_fallback)
            if is_valid(counts_fallback):
                best_A = A_fallback
                best_peak = get_peak(counts_fallback)
            else:
                # Absolute fallback
                return list(range(64, 129))

    # Phase 2: Local Search to minimize peak
    # Use best_A found
    A = best_A
    counts = build_counts(A)
    current_peak = get_peak(counts)
    
    # Local search loop
    no_improve_count = 0
    while time.time() - start_time < timeout:
        improved = False
        
        # Try removing elements
        A_list = list(A)
        random.shuffle(A_list)
        for a in A_list:
            if time.time() - start_time > timeout:
                break
            
            remove_element(A, counts, a)
            if is_valid(counts):
                new_peak = get_peak(counts)
                if new_peak < current_peak:
                    current_peak = new_peak
                    improved = True
                    no_improve_count = 0
                    continue
                elif new_peak == current_peak:
                    # Keep if same peak but smaller set? 
                    # Smaller set might allow future improvements.
                    improved = True
                    no_improve_count = 0
                    continue
                else:
                    add_element(A, counts, a)
            else:
                add_element(A, counts, a)
        
        # Try swapping elements
        if not improved:
            A_list = list(A)
            random.shuffle(A_list)
            for a in A_list:
                if time.time() - start_time > timeout:
                    break
                
                # Try swapping a with some b not in A
                # Sample a few candidates
                candidates = random.sample([x for x in range(n + 1) if x not in A], min(10, len([x for x in range(n + 1) if x not in A])))
                for b in candidates:
                    if time.time() - start_time > timeout:
                        break
                    
                    # Remove a
                    remove_element(A, counts, a)
                    # Add b
                    add_element(A, counts, b)
                    
                    if is_valid(counts):
                        new_peak = get_peak(counts)
                        if new_peak < current_peak:
                            current_peak = new_peak
                            improved = True
                            no_improve_count = 0
                            break # Restart swap search
                        else:
                            # Revert
                            remove_element(A, counts, b)
                            add_element(A, counts, a)
                    else:
                        # Revert
                        remove_element(A, counts, b)
                        add_element(A, counts, a)
                
                if improved:
                    break
        
        if not improved:
            no_improve_count += 1
            # Perturbation: add a random element, then try to remove others
            if no_improve_count > 5:
                # Add a random element to escape local minimum
                candidates = [x for x in range(n + 1) if x not in A]
                if candidates:
                    b = random.choice(candidates)
                    add_element(A, counts, b)
                    # Try to remove some elements now
                    A_list = list(A)
                    random.shuffle(A_list)
                    for a in A_list:
                        if time.time() - start_time > timeout:
                            break
                        remove_element(A, counts, a)
                        if is_valid(counts):
                            new_peak = get_peak(counts)
                            if new_peak <= current_peak:
                                current_peak = new_peak
                                no_improve_count = 0
                                break
                            else:
                                add_element(A, counts, a)
                        else:
                            add_element(A, counts, a)
                    no_improve_count = 0
                else:
                    break
        else:
            no_improve_count = 0

    return list(A)

# EVOLVE-BLOCK-END
