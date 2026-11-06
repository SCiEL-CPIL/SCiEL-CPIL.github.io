import time
import random
import math

def construct():
    start_time = time.time()
    # Use 85s to be safe against overhead and evaluation latency
    deadline = start_time + 85.0
    
    n = 256
    
    # Precompute 3-APs and 4-APs
    # Represent sets as bitmasks for fast intersection checks
    # Element i (1-indexed) corresponds to bit (i-1)
    
    aps3 = []
    aps4 = []
    
    # Map element to list of AP indices for fast delta calculation
    elem_to_aps3 = [[] for _ in range(n+1)]
    elem_to_aps4 = [[] for _ in range(n+1)]
    
    # Generate 3-APs
    ap_id = 0
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            if c <= n:
                mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
                aps3.append((a, b, c, mask))
                elem_to_aps3[a].append(ap_id)
                elem_to_aps3[b].append(ap_id)
                elem_to_aps3[c].append(ap_id)
                ap_id += 1
                
    # Generate 4-APs
    ap_id = 0
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
                aps4.append((a, b, c, e, mask))
                elem_to_aps4[a].append(ap_id)
                elem_to_aps4[b].append(ap_id)
                elem_to_aps4[c].append(ap_id)
                elem_to_aps4[e].append(ap_id)
                ap_id += 1

    def count_3aps(A_mask):
        count = 0
        for _, _, _, mask in aps3:
            if (A_mask & mask) == mask:
                count += 1
        return count
        
    def count_4aps(A_mask):
        count = 0
        for _, _, _, _, mask in aps4:
            if (A_mask & mask) == mask:
                count += 1
        return count

    def is_valid_add(A_mask, x):
        # Check if adding x creates a 4-AP
        # x is 1-indexed
        bit_x = 1 << (x-1)
        for ap_idx in elem_to_aps4[x]:
            a, b, c, d, mask = aps4[ap_idx]
            others_mask = mask ^ bit_x
            if (A_mask & others_mask) == others_mask:
                return False
        return True

    def get_3ap_gain(A_mask, x):
        """Calculate how many new 3-APs are formed if x is added to A_mask."""
        bit_x = 1 << (x-1)
        gained = 0
        for ap_idx in elem_to_aps3[x]:
            a, b, c, mask = aps3[ap_idx]
            others_mask = mask ^ bit_x
            if (A_mask & others_mask) == others_mask:
                gained += 1
        return gained

    def get_3ap_loss(A_mask, x):
        """Calculate how many 3-APs are broken if x is removed from A_mask."""
        bit_x = 1 << (x-1)
        lost = 0
        for ap_idx in elem_to_aps3[x]:
            a, b, c, mask = aps3[ap_idx]
            others_mask = mask ^ bit_x
            if (A_mask & others_mask) == others_mask:
                lost += 1
        return lost

    def run_greedy_init(seed):
        """
        Greedy initialization that prioritizes elements with high 3-AP potential.
        """
        rng = random.Random(seed)
        A_mask = 0
        A_set = set()
        
        # Sort elements by "potential" - initially just random, but we can refine
        # A better heuristic: Start with a random subset, then add best valid elements
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        
        # Phase 1: Random valid addition to get a base
        for x in candidates:
            if is_valid_add(A_mask, x):
                A_mask |= (1 << (x-1))
                A_set.add(x)
                
        # Phase 2: Try to improve by swapping or adding high-gain elements
        # We do a few passes of "best valid add"
        for _ in range(3):
            improved = False
            best_gain = 0
            best_x = None
            for x in range(1, n + 1):
                if x in A_set:
                    continue
                if not is_valid_add(A_mask, x):
                    continue
                gain = get_3ap_gain(A_mask, x)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is not None and best_gain > 0:
                A_mask |= (1 << (best_x-1))
                A_set.add(best_x)
                improved = True
            if not improved:
                break
                
        return A_mask, A_set

    def run_sa(init_mask, init_set, time_limit, seed):
        """
        Simulated Annealing to refine the set.
        Moves: Add valid element, Remove element, Swap element.
        Objective: Maximize 3-APs, subject to 0 4-APs.
        """
        rng = random.Random(seed)
        A_mask = init_mask
        A_set = init_set
        current_score = count_3aps(A_mask)
        
        best_mask = A_mask
        best_set = A_set
        best_score = current_score
        
        # SA Parameters
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.95
        
        # We will run until time_limit is reached
        steps = 0
        while time.time() < time_limit:
            # Generate a random move
            move_type = rng.random()
            
            new_mask = A_mask
            new_set = A_set
            delta_score = 0
            valid = True
            
            if move_type < 0.3:
                # Add a random element
                # Find valid candidates
                candidates = [x for x in range(1, n+1) if x not in A_set and is_valid_add(A_mask, x)]
                if not candidates:
                    continue
                x = rng.choice(candidates)
                new_mask |= (1 << (x-1))
                new_set = A_set | {x}
                delta_score = get_3ap_gain(A_mask, x)
                
            elif move_type < 0.6:
                # Remove a random element
                if not A_set:
                    continue
                x = rng.choice(list(A_set))
                new_mask &= ~(1 << (x-1))
                new_set = A_set - {x}
                delta_score = -get_3ap_loss(A_mask, x)
                
            else:
                # Swap: Remove u, Add v
                if not A_set:
                    continue
                u = rng.choice(list(A_set))
                # Find v such that (A \ {u}) U {v} is valid
                # This is slightly more complex to check directly.
                # Let's try a random v not in A, and check validity of the new set.
                v = rng.randint(1, n)
                if v == u or v in A_set:
                    continue
                
                # Check if adding v to A\{u} is valid
                # Temporarily remove u
                mask_no_u = A_mask & ~(1 << (u-1))
                if is_valid_add(mask_no_u, v):
                    new_mask = mask_no_u | (1 << (v-1))
                    new_set = (A_set - {u}) | {v}
                    
                    # Calculate delta
                    # Lost: 3-APs involving u that were in A
                    lost = get_3ap_loss(A_mask, u)
                    # Gained: 3-APs involving v in new set (A\{u})
                    gained = get_3ap_gain(mask_no_u, v)
                    # Note: get_3ap_gain(mask_no_u, v) counts 3-APs formed by v with elements in mask_no_u.
                    # This is correct for the new state.
                    
                    delta_score = gained - lost
                else:
                    continue
            
            if not valid:
                continue
            
            new_score = current_score + delta_score
            
            # Acceptance probability
            if new_score > current_score:
                accept = True
            else:
                # Accept worse solution with probability exp(-delta / T)
                # delta is negative here
                prob = math.exp(delta_score / T)
                accept = rng.random() < prob
            
            if accept:
                A_mask = new_mask
                A_set = new_set
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = A_mask
                    best_set = A_set
                    
            T *= cooling_rate
            if T < T_min:
                # Restart temperature occasionally or break if stuck
                # For simplicity, we just let it cool. 
                # If T is too low, we might restart in the outer loop.
                T = T_min
            
            steps += 1
            
            # Check time frequently
            if time.time() > time_limit:
                break
                
        return best_mask, best_set, best_score

    # Main logic
    best_mask = 0
    best_set = set()
    best_score = 0
    
    # Run multiple SA instances with different seeds and initializations
    num_runs = 10
    time_per_run = (deadline - time.time()) / num_runs
    
    for i in range(num_runs):
        if time.time() > deadline - 2:
            break
            
        # Initialize
        init_mask, init_set = run_greedy_init(i)
        
        # Run SA with a time budget
        # We subtract a small buffer from time_per_run to ensure we don't overshoot
        sa_limit = time.time() + max(0.1, time_per_run * 0.9)
        
        sa_mask, sa_set, sa_score = run_sa(init_mask, init_set, sa_limit, i)
        
        if sa_score > best_score:
            best_score = sa_score
            best_mask = sa_mask
            best_set = sa_set
            
    # Final verification
    if count_4aps(best_mask) != 0:
        # Fallback to a known valid set if something went wrong (shouldn't happen with strict validity checks)
        # Just use the first greedy init
        A_mask, A_set = run_greedy_init(0)
        best_mask = A_mask
        best_set = A_set
        best_score = count_3aps(best_mask)

    return sorted(best_set)
