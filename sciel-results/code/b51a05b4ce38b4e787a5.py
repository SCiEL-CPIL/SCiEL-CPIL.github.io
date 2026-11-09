"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin before 90s kill
    
    N = 512
    
    # Precompute 4-APs
    # ap_triples[i] contains lists of (j, k, l) such that (i, j, k, l) is a 4-AP
    # Note: The AP is unordered in terms of "which one is missing", but ordered in value.
    # For a given i, we want to know which triples (j, k, l) form an AP with i.
    ap_triples = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            ap_triples[a].append((b, c, e))
            ap_triples[b].append((a, c, e))
            ap_triples[c].append((a, b, e))
            ap_triples[e].append((a, b, c))

    # Helper to check if a set is valid (should not be needed if we maintain state correctly, 
    # but good for initialization)
    def is_valid(selected):
        for a in range(1, N + 1):
            if selected[a]:
                for d in range(1, (N - a) // 3 + 1):
                    b = a + d
                    c = a + 2 * d
                    e = a + 3 * d
                    if e > N: break
                    if selected[b] and selected[c] and selected[e]:
                        return False
        return True

    # Simulated Annealing State
    # selected: boolean array 1..N
    # missing_count[i]: number of APs containing i where the other 3 are in 'selected'
    # If missing_count[i] == 0, we can add i.
    
    def init_state():
        selected = [False] * (N + 1)
        missing_count = [0] * (N + 1)
        # Start with empty set
        return selected, missing_count, 0

    def can_add(selected, missing_count, i):
        return missing_count[i] == 0

    def add_element(selected, missing_count, i):
        # Precondition: can_add is true
        selected[i] = True
        count = 1
        # Update missing_count for neighbors
        for (j, k, l) in ap_triples[i]:
            # The AP is (i, j, k, l) in some order? No, ap_triples[i] stores (other, other, other).
            # Let's say the AP is (i, j, k, l).
            # If j, k, l are in selected, then missing_count[i] would have been > 0.
            # But we are adding i.
            # For j: The AP is (j, i, k, l). The other three are i, k, l.
            # If k and l are in selected, then missing_count[j] increases by 1.
            if selected[k] and selected[l]:
                missing_count[j] += 1
            # For k: The AP is (k, i, j, l). The other three are i, j, l.
            # If j and l are in selected, then missing_count[k] increases by 1.
            if selected[j] and selected[l]:
                missing_count[k] += 1
            # For l: The AP is (l, i, j, k). The other three are i, j, k.
            # If j and k are in selected, then missing_count[l] increases by 1.
            if selected[j] and selected[k]:
                missing_count[l] += 1
        return count

    def remove_element(selected, missing_count, i):
        # Precondition: selected[i] is True
        selected[i] = False
        # Update missing_count for neighbors
        for (j, k, l) in ap_triples[i]:
            if selected[k] and selected[l]:
                missing_count[j] -= 1
            if selected[j] and selected[l]:
                missing_count[k] -= 1
            if selected[j] and selected[k]:
                missing_count[l] -= 1
        return -1

    # Initial Solution: Greedy
    def greedy_init():
        selected, missing_count, count = init_state()
        # Order by random or conflict
        order = list(range(1, N + 1))
        random.shuffle(order)
        for i in order:
            if can_add(selected, missing_count, i):
                add_element(selected, missing_count, i)
                count += 1
        return selected, missing_count, count

    best_selected, best_missing, best_count = greedy_init()
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-4
    
    # To speed up, we can try multiple restarts or just run SA for a long time.
    # Given 88s, we can run SA for a significant number of steps.
    
    # We need a fast way to pick a random element to add/remove/swap.
    # Let's stick to simple moves:
    # 1. Add a random i with missing_count[i] == 0.
    # 2. Remove a random i with selected[i] == True.
    # 3. Swap: Remove i, Add j. (Requires careful state update).
    
    # To make it faster, we can maintain a list of "addable" elements.
    # But maintaining that list dynamically is overhead.
    # Let's just pick random indices. N=512 is small.
    
    current_selected, current_missing, current_count = best_selected[:], best_missing[:], best_count
    
    steps = 0
    max_steps = 5000000 # Adjust based on speed
    
    while time.time() < deadline - 2.0:
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add
            i = random.randint(1, N)
            if not current_selected[i] and can_add(current_selected, current_missing, i):
                add_element(current_selected, current_missing, i)
                current_count += 1
                # Always accept if size increases
                if current_count > best_count:
                    best_count = current_count
                    best_selected = current_selected[:]
                    best_missing = current_missing[:]
            elif not current_selected[i]:
                # If we can't add, maybe we accept a "bad" move? No, constraint is hard.
                # So we just skip or try another.
                pass
        elif move_type < 0.7:
            # Try to remove
            i = random.randint(1, N)
            if current_selected[i]:
                remove_element(current_selected, current_missing, i)
                current_count -= 1
                # Accept with probability exp(-delta/T) where delta = -1 (size decrease)
                # Actually, we want to maximize size. Delta in objective (size) is -1.
                # We accept if random < exp(-(-1)/T) = exp(1/T). This is always 1 if T is small?
                # No, standard SA minimizes energy. Here we maximize size.
                # Let's define Energy = -Size.
                # Move: Size decreases by 1. Energy increases by 1.
                # Accept if random < exp(-1/T).
                if random.random() < math.exp(-1.0 / T):
                    pass # Keep the move
                else:
                    # Revert
                    add_element(current_selected, current_missing, i)
                    current_count += 1
        else:
            # Swap: Remove i, Add j
            i = random.randint(1, N)
            j = random.randint(1, N)
            if i == j: continue
            if current_selected[i] and not current_selected[j]:
                # Remove i
                remove_element(current_selected, current_missing, i)
                current_count -= 1
                # Check if j can be added
                if can_add(current_selected, current_missing, j):
                    add_element(current_selected, current_missing, j)
                    current_count += 1
                    # Net change 0 in size.
                    # Why swap? To explore different configurations.
                    # Accept based on diversity? Or just accept?
                    # If we accept all swaps, we might wander.
                    # Let's accept with high probability to explore.
                    if random.random() < 0.8:
                         pass
                    else:
                         # Revert
                         remove_element(current_selected, current_missing, j)
                         current_count -= 1
                         add_element(current_selected, current_missing, i)
                         current_count += 1
                else:
                    # Revert
                    add_element(current_selected, current_missing, i)
                    current_count += 1
            
        # Cooling
        T *= cooling_rate
        if T < min_T:
            T = min_T
            # Maybe restart with a higher T?
            # Or just stop if stuck?
            # Let's restart SA with a fresh greedy if T is too low and no improvement for a while.
            # For now, just keep going.
            
        steps += 1

    result = [i for i in range(1, N + 1) if best_selected[i]]
    return result

# EVOLVE-BLOCK-END
