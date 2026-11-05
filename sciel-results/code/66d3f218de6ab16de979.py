"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 512
    
    # Precompute 4-AP conflict masks
    x_conflicts = [[] for _ in range(n+1)]
    
    for d in range(1, 171):
        for a in range(1, 512 - 3*d + 1):
            terms = [a, a+d, a+2*d, a+3*d]
            for pos in range(4):
                x = terms[pos]
                others = [terms[j] for j in range(4) if j != pos]
                mask = 0
                for y in others:
                    mask |= (1 << (y-1))
                x_conflicts[x].append(mask)
    
    # Greedy initial solution
    S = 0
    for x in range(1, n+1):
        bit = 1 << (x-1)
        conflict = False
        for mask in x_conflicts[x]:
            if (S & mask) == mask:
                conflict = True
                break
        if not conflict:
            S |= bit
    
    best_S = S
    best_count = bin(S).count('1')
    
    # Local search with periodic fill passes (inspired by improvement 1.104 -> 1.112)
    rng = random.Random(42)
    T = 5.0
    T_min = 0.01
    cooling = 0.9999
    fill_interval = 200  # do a fill pass every 200 iterations
    
    iteration = 0
    while time.time() < deadline - 3.0 and T > T_min:
        if iteration % fill_interval == 0 and time.time() < deadline - 3.0:
            # Periodic fill pass: try to add any element that can be added
            for x in range(1, n+1):
                bit = 1 << (x-1)
                if not (S & bit):
                    conflict = False
                    for mask in x_conflicts[x]:
                        if (S & mask) == mask:
                            conflict = True
                            break
                    if not conflict:
                        S |= bit
            current_count = bin(S).count('1')
            if current_count > best_count:
                best_count = current_count
                best_S = S
        
        # Pick a random move
        x = rng.randint(1, n)
        bit = 1 << (x-1)
        
        if S & bit:
            # Try removing x
            S = S & ~bit
        else:
            # Try adding x
            conflict = False
            for mask in x_conflicts[x]:
                if (S & mask) == mask:
                    conflict = True
                    break
            if not conflict:
                S |= bit
            else:
                # Try a swap: remove a random conflicting element and add x
                # Find elements that conflict with x
                conflicting_elements = set()
                for mask in x_conflicts[x]:
                    if (S & mask) == mask:
                        # Extract bits from mask that are in S
                        temp = mask & S
                        while temp:
                            lsb = temp & (-temp)
                            idx = lsb.bit_length()
                            conflicting_elements.add(idx)
                            temp ^= lsb
                
                if conflicting_elements:
                    # Try removing one random conflicting element and adding x
                    victim = rng.choice(list(conflicting_elements))
                    victim_bit = 1 << (victim-1)
                    
                    # Check if after removing victim, x can be added
                    temp_S = S & ~victim_bit
                    can_add = True
                    for mask in x_conflicts[x]:
                        if (temp_S & mask) == mask:
                            can_add = False
                            break
                    
                    if can_add:
                        S = temp_S | bit
                    else:
                        # Just do the removal
                        S = temp_S
        
        current_count = bin(S).count('1')
        if current_count > best_count:
            best_count = current_count
            best_S = S
        
        # Simulated annealing acceptance: sometimes accept worse moves
        if current_count < best_count and rng.random() < math.exp(-(best_count - current_count) / T):
            pass  # Already applied the move
        
        T *= cooling
        iteration += 1
    
    # Final fill pass
    for x in range(1, n+1):
        bit = 1 << (x-1)
        if not (best_S & bit):
            conflict = False
            for mask in x_conflicts[x]:
                if (best_S & mask) == mask:
                    conflict = True
                    break
            if not conflict:
                best_S |= bit
    
    # Convert bitmask to sorted list
    result = [x for x in range(1, n+1) if best_S & (1 << (x-1))]
    return result

# EVOLVE-BLOCK-END
