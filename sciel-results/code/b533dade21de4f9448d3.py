import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 256
    
    # Precompute 3-APs and 4-APs
    # Represent sets as integers where bit i corresponds to element i (1-indexed, so bit 0 is unused or bit 1 is element 1)
    # Let's use bit i for element i+1 to make bit 0 element 1? 
    # Standard: bit i represents element i. Let's use 1-based indexing in logic, so element 1 is bit 1.
    # Mask for element x is (1 << x)
    
    three_aps = []
    four_aps = []
    
    for a in range(1, N + 1):
        max_d_3 = (N - a) // 2
        max_d_4 = (N - a) // 3
        for d in range(1, max_d_3 + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                mask = (1 << a) | (1 << b) | (1 << c)
                three_aps.append(mask)
        
        for d in range(1, max_d_4 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                mask = (1 << a) | (1 << b) | (1 << c) | (1 << e)
                four_aps.append(mask)
    
    # Precompute adjacency for fast delta calculation
    # ap3_by_elem[x] = list of (other1, other2, ap_index)
    ap3_by_elem = [[] for _ in range(N + 1)]
    for idx, mask in enumerate(three_aps):
        # Extract elements
        # This is slow if done generically. Let's just store the tuple (a,b,c)
        pass

    # Better precomputation:
    # Store (a, b, c) for 3-APs and (a, b, c, e) for 4-APs
    three_aps_data = []
    four_aps_data = []
    
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                three_aps_data.append((a, b, c))
                
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                four_aps_data.append((a, b, c, e))
    
    # Build indices
    ap3_indices = [[] for _ in range(N + 1)]
    for idx, (a, b, c) in enumerate(three_aps_data):
        ap3_indices[a].append(idx)
        ap3_indices[b].append(idx)
        ap3_indices[c].append(idx)
        
    ap4_indices = [[] for _ in range(N + 1)]
    for idx, (a, b, c, e) in enumerate(four_aps_data):
        ap4_indices[a].append(idx)
        ap4_indices[b].append(idx)
        ap4_indices[c].append(idx)
        ap4_indices[e].append(idx)
        
    # Helper to check if adding x creates a 4-AP
    def creates_4ap(mask, x):
        for idx in ap4_indices[x]:
            a, b, c, e = four_aps_data[idx]
            # Check if the other 3 are in mask
            # We can do this by checking bits
            other_mask = 0
            for y in (a, b, c, e):
                if y != x:
                    other_mask |= (1 << y)
            if (mask & other_mask) == other_mask:
                return True
        return False
        
    # Helper to count new 3-APs created by adding x
    def gain_3aps(mask, x):
        gain = 0
        for idx in ap3_indices[x]:
            a, b, c = three_aps_data[idx]
            other_mask = 0
            for y in (a, b, c):
                if y != x:
                    other_mask |= (1 << y)
            if (mask & other_mask) == other_mask:
                gain += 1
        return gain
        
    # Helper to count 3-APs lost by removing x
    def loss_3aps(mask, x):
        loss = 0
        for idx in ap3_indices[x]:
            a, b, c = three_aps_data[idx]
            other_mask = 0
            for y in (a, b, c):
                if y != x:
                    other_mask |= (1 << y)
            if (mask & other_mask) == other_mask:
                loss += 1
        return loss
        
    # Greedy Initialization
    def greedy_init():
        mask = 0
        # Start with random subset or empty? 
        # Let's try to build a dense set.
        # We will add elements in random order if they don't create 4-APs and have high gain.
        
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        # First pass: add if no 4-AP
        for x in candidates:
            if not creates_4ap(mask, x):
                mask |= (1 << x)
                
        # Second pass: try to add remaining if they don't create 4-AP (maybe order matters)
        # Actually, greedy is order dependent. Let's do a simple greedy with random restarts for init.
        
        return mask
        
    # Simulated Annealing
    def simulated_annealing(initial_mask, steps=50000):
        current_mask = initial_mask
        current_3ap_count = 0
        
        # Calculate initial count
        for a, b, c in three_aps_data:
            if (current_mask & (1 << a)) and (current_mask & (1 << b)) and (current_mask & (1 << c)):
                current_3ap_count += 1
                
        best_mask = current_mask
        best_3ap_count = current_3ap_count
        
        T = 10.0
        cooling_rate = 0.9995
        min_T = 0.1
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            if T < min_T:
                break
                
            # Generate move
            # 1. Add random element
            # 2. Remove random element
            # 3. Swap
            
            move_type = random.random()
            
            if move_type < 0.3:
                # Add
                # Pick random x not in mask
                while True:
                    x = random.randint(1, N)
                    if not (current_mask & (1 << x)):
                        break
                        # If all are in mask, break
                    if x == 1 and (current_mask == (1 << (N+1)) - 2): # All bits set except 0
                         break
                 
                if creates_4ap(current_mask, x):
                    continue
                 
                delta = gain_3aps(current_mask, x)
                new_mask = current_mask | (1 << x)
                new_count = current_3ap_count + delta
                
                if new_count > best_3ap_count:
                    best_mask = new_mask
                    best_3ap_count = new_count
                    
                # Acceptance
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_3ap_count = new_count
                    
            elif move_type < 0.6:
                # Remove
                # Pick random x in mask
                # List of set bits is slow. 
                # We can pick random x and check if in mask.
                while True:
                    x = random.randint(1, N)
                    if (current_mask & (1 << x)):
                        break
                    if x == 1 and current_mask == 0:
                        break
                if current_mask == 0:
                    break
                    
                delta = -loss_3aps(current_mask, x)
                new_mask = current_mask & ~(1 << x)
                new_count = current_3ap_count + delta
                
                if new_count > best_3ap_count:
                    best_mask = new_mask
                    best_3ap_count = new_count
                    
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_3ap_count = new_count
                    
            else:
                # Swap
                # Pick x_in in mask, x_out not in mask
                while True:
                    x_in = random.randint(1, N)
                    if (current_mask & (1 << x_in)):
                        break
                while True:
                    x_out = random.randint(1, N)
                    if not (current_mask & (1 << x_out)):
                        break
                if current_mask == 0 or current_mask == (1 << (N+1)) - 2:
                     break

                # Check if removing x_in and adding x_out is valid
                # 1. Remove x_in
                temp_mask = current_mask & ~(1 << x_in)
                # 2. Check if adding x_out to temp_mask creates 4-AP
                if creates_4ap(temp_mask, x_out):
                    continue
                
                # Calculate delta
                loss = loss_3aps(current_mask, x_in)
                gain = gain_3aps(temp_mask, x_out)
                delta = gain - loss
                
                new_mask = temp_mask | (1 << x_out)
                new_count = current_3ap_count + delta
                
                if new_count > best_3ap_count:
                    best_mask = new_mask
                    best_3ap_count = new_count
                    
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_3ap_count = new_count
                    
            T *= cooling_rate
            
        return best_mask, best_3ap_count
        
    # Main Logic
    best_overall_mask = 0
    best_overall_count = 0
    
    # Run multiple restarts
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        # Try different strategies
        if i == 0:
            mask = greedy_init()
        else:
            # Random valid set
            mask = 0
            # Add random elements
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            for x in candidates:
                if not creates_4ap(mask, x):
                    mask |= (1 << x)
                    
        # Run SA
        res_mask, res_count = simulated_annealing(mask, steps=10000)
        
        if res_count > best_overall_count:
            best_overall_count = res_count
            best_overall_mask = res_mask
            
        # Also try local search on the best found so far
        # The SA already does this, but a dedicated hill climb might polish it
        # Let's just rely on SA with enough steps
        
    # Convert mask to sorted list
    result = []
    for i in range(1, N + 1):
        if best_overall_mask & (1 << i):
            result.append(i)
            
    if not result:
        # Fallback
        # Find a small valid set
        # {1, 2, 4, 8...} powers of 2 are 4-AP free?
        # 1, 2, 4, 8: 1,2,4,8 is not AP.
        # 1, 3, 5, 7 is 4-AP.
        # Let's just return {1} if empty
        result = [1]
        
    return result
