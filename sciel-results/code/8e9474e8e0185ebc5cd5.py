import time
import random

N = 256
DEADLINE = 55.0

def _precompute():
    # 0-indexed arrays of size N
    # ap3_partners[i] is a list of (j, k) such that {i, j, k} forms a 3-AP
    # ap4_partners[i] is a list of (j, k, l) such that {i, j, k, l} forms a 4-AP
    ap3_partners = [[] for _ in range(N)]
    ap4_partners = [[] for _ in range(N)]
    
    # 3-APs: (a, a+d, a+2d)
    for d in range(1, N // 2 + 1):
        for a in range(N - 2 * d):
            b = a + d
            c = a + 2 * d
            ap3_partners[a].append((b, c))
            ap3_partners[b].append((a, c))
            ap3_partners[c].append((a, b))
            
    # 4-APs: (a, a+d, a+2d, a+3d)
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap4_partners[a].append((b, c, e))
            ap4_partners[b].append((a, c, e))
            ap4_partners[c].append((a, b, e))
            ap4_partners[e].append((a, b, c))
            
    return ap3_partners, ap4_partners

AP3_PARTNERS, AP4_PARTNERS = _precompute()

def _count_3aps(mask):
    total = 0
    for i in range(N):
        if (mask >> i) & 1:
            for j, k in AP3_PARTNERS[i]:
                if (mask >> j) & 1 and (mask >> k) & 1:
                    total += 1
    return total // 3

def _is_valid_add(x, mask):
    for j, k, l in AP4_PARTNERS[x]:
        if (mask & (1 << j)) and (mask & (1 << k)) and (mask & (1 << l)):
            return False
    return True

def _is_valid_swap(x_out, x_in, mask):
    # Check if x_in can be added to mask \ {x_out}
    temp_mask = mask & ~(1 << x_out)
    return _is_valid_add(x_in, temp_mask)

def _calculate_gain(x, mask, exclude=None):
    count = 0
    for j, k in AP3_PARTNERS[x]:
        if exclude is not None:
            j_ok = (j != exclude) and (mask & (1 << j))
            k_ok = (k != exclude) and (mask & (1 << k))
        else:
            j_ok = (mask & (1 << j))
            k_ok = (mask & (1 << k))
        if j_ok and k_ok:
            count += 1
    return count

def _calculate_loss(x, mask):
    return _calculate_gain(x, mask)

def _init_greedy(rng, randomize_top_k=5):
    mask = 0
    remaining = list(range(N))
    rng.shuffle(remaining)
    
    while remaining:
        valid = []
        gains = {}
        for x in remaining:
            if _is_valid_add(x, mask):
                g = _calculate_gain(x, mask)
                valid.append(x)
                gains[x] = g
        
        if not valid:
            break
            
        # Sort by gain descending
        valid.sort(key=lambda x: gains[x], reverse=True)
        
        # Pick from top K
        K = min(randomize_top_k, len(valid))
        x = rng.choice(valid[:K])
        
        mask |= (1 << x)
        remaining.remove(x)
        
    return mask

def _walksat(mask, deadline, rng):
    current_mask = mask
    current_score = _count_3aps(current_mask)
    best_mask = current_mask
    best_score = current_score
    
    present = [i for i in range(N) if (current_mask >> i) & 1]
    absent = [i for i in range(N) if not ((current_mask >> i) & 1)]
    
    # WalkSAT parameters
    # We want to maximize 3-APs while keeping 4-APs at 0.
    # This is a Max-SAT problem where:
    # Hard constraints: No 4-AP.
    # Soft constraints: Maximize 3-APs.
    
    # In WalkSAT, we usually flip variables to satisfy clauses.
    # Here, "violating a hard constraint" is bad. "Satisfying a soft constraint" is good.
    
    # Strategy:
    # 1. If current state is invalid (has 4-APs), prioritize removing elements to fix it.
    # 2. If valid, try to improve score.
    #    - With probability p_flip, flip a random variable (add/remove/swap) to escape local optima.
    #    - Otherwise, do a greedy step (best improving move).
    
    # To simplify, we stick to valid states only (since removing fixes 4-APs, and we check validity for adds/swaps).
    # So we are always in the "satisfied hard constraints" space.
    # We just want to maximize soft constraints.
    
    # However, sometimes we might be stuck.
    # A "flip" in this context could be a random move that doesn't necessarily improve score, 
    # but must maintain validity.
    
    # Let's use a hybrid:
    # - 80% of the time: Try all improving moves (Add, Remove, Swap). Pick best.
    # - 20% of the time: Try a random valid move (Add, Remove, Swap).
    
    # This is essentially Simulated Annealing with T=0 for 80% and T=infinity for 20%?
    # Or just random walk.
    
    # Let's implement a standard WalkSAT-like approach on the valid subgraph.
    # Actually, since we can't violate hard constraints, we can't really "walk" in the invalid space.
    # So we are doing local search on the valid states.
    
    # To escape local optima, we can do "restarts" or "random walks".
    # Random walk: Pick a random valid move (even if it decreases score).
    
    # Let's combine:
    # - If stuck (no improving move), do a random valid move.
    # - Or always mix.
    
    # Let's try:
    # 1. Generate all valid moves and their deltas.
    # 2. If there are improving moves, pick the best one (or random from top few).
    # 3. If no improving moves, pick a random valid move (even if negative delta).
    
    # To make it faster, we don't generate ALL moves every time if N is large, but N=256 is small enough.
    # Number of moves:
    # Adds: ~N
    # Removes: ~N
    # Swaps: ~N^2 (too many to check all every step).
    # So we restrict swaps to random samples.
    
    while time.time() < deadline:
        improving_moves = []
        all_valid_moves = []
        
        # 1. Check Adds
        for x in absent:
            if _is_valid_add(x, current_mask):
                delta = _calculate_gain(x, current_mask)
                move = ('add', x, None)
                all_valid_moves.append((delta, move))
                if delta > 0:
                    improving_moves.append((delta, move))
        
        # 2. Check Removes
        for x in present:
            delta = -_calculate_loss(x, current_mask)
            move = ('remove', x, None)
            all_valid_moves.append((delta, move))
            if delta > 0:
                improving_moves.append((delta, move))
        
        # 3. Check Swaps (Sampled)
        # Sample 10 present and 10 absent
        if present and absent:
            sample_p = rng.sample(present, min(10, len(present)))
            sample_a = rng.sample(absent, min(10, len(absent)))
            for x_out in sample_p:
                for x_in in sample_a:
                    if _is_valid_swap(x_out, x_in, current_mask):
                        temp_mask = current_mask & ~(1 << x_out)
                        gain = _calculate_gain(x_in, temp_mask)
                        loss = _calculate_loss(x_out, current_mask)
                        delta = gain - loss
                        move = ('swap', x_out, x_in)
                        all_valid_moves.append((delta, move))
                        if delta > 0:
                            improving_moves.append((delta, move))
        
        # Decide next move
        if improving_moves:
            # Pick best improving move
            # To avoid getting stuck in a specific local optimum, pick randomly from top 3?
            improving_moves.sort(key=lambda x: x[0], reverse=True)
            top_k = min(3, len(improving_moves))
            _, move = rng.choice(improving_moves[:top_k])
        else:
            # No improving moves. Pick a random valid move to escape.
            if not all_valid_moves:
                break # No moves at all? Should not happen unless set is empty/full and no swaps possible
            _, move = rng.choice(all_valid_moves)
        
        # Apply move
        if move[0] == 'add':
            x = move[1]
            current_mask |= (1 << x)
            absent.remove(x)
            present.append(x)
        elif move[0] == 'remove':
            x = move[1]
            current_mask &= ~(1 << x)
            present.remove(x)
            absent.append(x)
        elif move[0] == 'swap':
            x_out, x_in = move[1], move[2]
            current_mask &= ~(1 << x_out)
            current_mask |= (1 << x_in)
            present.remove(x_out)
            present.append(x_in)
            absent.remove(x_in)
            absent.append(x_out)
        
        # Update score
        # We could track delta, but recalculating is safer and fast enough for N=256
        # _count_3aps is O(N * deg) ~ 256 * 50 ~ 12k ops. Very fast.
        current_score = _count_3aps(current_mask)
        
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
            
    return best_mask, best_score

def construct():
    start_time = time.time()
    rng = random.Random() # Use system time for variety, or fixed?
    # The prompt implies we should maximize. Using random seed allows different runs to find different optima.
    # However, for a single evaluation, we just need to be good.
    # Let's use a fixed seed for reproducibility in this specific file, but the evaluator might run it once.
    # Actually, "Maximize" usually means the program should find the best it can.
    # If I use a fixed seed, I get the same result.
    # If I use random, I might get lucky.
    # Let's use a fixed seed to be deterministic in the code, but maybe multiple restarts?
    # The budget is 60s.
    
    # Strategy:
    # 1. Generate a few diverse initial solutions using GRASP.
    # 2. Run WalkSAT/Local Search on the best one for the remaining time.
    # Or, run WalkSAT in parallel? No, single process.
    # Run WalkSAT on the best initial solution.
    
    # Generate 10 initial solutions
    inits = []
    for i in range(10):
        # Vary the randomness in greedy
        mask = _init_greedy(rng, randomize_top_k=3 + i)
        inits.append(mask)
        
    # Evaluate and pick best
    best_init = 0
    best_init_score = -1
    for m in inits:
        s = _count_3aps(m)
        if s > best_init_score:
            best_init_score = s
            best_init = m
            
    # Run WalkSAT from best init
    # Leave 5 seconds for safety
    deadline = start_time + 50.0
    final_mask, final_score = _walksat(best_init, deadline, rng)
    
    # Convert mask to list of 1-based indices
    result = [i + 1 for i in range(N) if (final_mask >> i) & 1]
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Size: {len(res)}")
    # Verify no 4-APs
    mask = 0
    for x in res:
        mask |= (1 << (x-1))
    
    # Check 4-APs
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if (mask & (1 << a)) and (mask & (1 << b)) and (mask & (1 << c)) and (mask & (1 << e)):
                print(f"4-AP found: {a+1}, {b+1}, {c+1}, {e+1}")
                break
    
    # Count 3-APs
    print(f"3-APs: {_count_3aps(mask)}")
