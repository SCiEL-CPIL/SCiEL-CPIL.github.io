import time
import random
import numpy as np

def _is_sidon(arr):
    """Check if arr is a Sidon set (all pair sums a+b with a<=b are unique)."""
    n = len(arr)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _diffs(arr):
    """Return set of nonzero absolute differences |b-a| for a != b in arr."""
    d = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            d.add(abs(arr[j] - arr[i]))
    return d

def _score_sets(A, B):
    """
    Calculate a score for the pair of sets.
    Higher is better.
    We want to maximize size, but we must ensure validity.
    Score = (|A| + |B|) * 100 - (invalidity_penalty)
    If invalid, penalty is large.
    """
    size = len(A) + len(B)
    
    # Check Sidon for A
    if not _is_sidon(A):
        # Count violations roughly or just return low score
        # For speed in SA, we need fast penalty.
        # Let's count duplicate sums.
        sums = []
        n = len(A)
        for i in range(n):
            for j in range(i, n):
                sums.append(A[i] + A[j])
        dup_a = len(sums) - len(set(sums))
        if dup_a > 0:
            return -1000 - dup_a
            
    # Check Sidon for B
    if not _is_sidon(B):
        sums = []
        n = len(B)
        for i in range(n):
            for j in range(i, n):
                sums.append(B[i] + B[j])
        dup_b = len(sums) - len(set(sums))
        if dup_b > 0:
            return -1000 - dup_b
            
    # Check Diff Disjointness
    dA = _diffs(A)
    dB = _diffs(B)
    overlap = len(dA & dB)
    
    if overlap > 0:
        return -1000 - overlap * 10 + size
        
    return size * 100

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    # Initial population: Random valid-ish sets? 
    # It's hard to start with valid sets. 
    # Start with small random sets and grow? 
    # Or start with empty and insert? That's greedy.
    # Let's start with a random subset of [1, 256] and repair/optimize.
    
    best_A = []
    best_B = []
    best_score = -10000
    
    # Try a few initial random starts
    for _ in range(5):
        if time.time() > deadline:
            break
        # Random start: pick ~10-20 random numbers
        size_init = random.randint(10, 20)
        A = random.sample(range(1, 257), size_init)
        B = random.sample(range(1, 257), size_init)
        
        # Quick repair: remove elements causing Sidon violations or diff overlaps
        # This is complex. Instead, let's just let SA handle the penalty.
        # But if penalty is too high, it might get stuck.
        
        # Let's try a "Greedy Seed" then SA.
        # The current best program (1.35) uses greedy restarts.
        # My SA will take that result and try to improve it via swaps.
        
        pass # We will initialize from a greedy run in the main loop below.

    # Phase 1: Get a good baseline using the existing greedy logic (fast)
    # Re-implementing a fast greedy here to seed the SA
    def _greedy_seed():
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        sums_A = set()
        sums_B = set()
        
        # We need to maintain sum sets for O(1) Sidon check on insertion
        # But sums change when we insert.
        # For a quick seed, let's just use the O(n^2) check as in the prompt's current code, 
        # but limited to a few restarts.
        
        best_seed = ([], [])
        best_seed_size = 0
        
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            order = list(range(1, 257))
            random.shuffle(order)
            
            cur_A = []
            cur_B = []
            dA = set()
            dB = set()
            
            for x in order:
                # Try adding to A
                valid_A = True
                # Sidon check for A + [x]
                # New sums: x+x, x+a for a in A
                # Check against existing sums in A
                # We don't maintain sum set in this fast loop for simplicity of code structure,
                # but we can do a quick check.
                # Actually, let's just use the full check for the seed since it's only 20 restarts.
                
                cand_A = cur_A + [x]
                if _is_sidon(cand_A):
                    new_dA = set(abs(x - a) for a in cur_A)
                    if not (new_dA & dB):
                        # Valid for A
                        # Check if we should prefer A or B?
                        # Just pick randomly or based on size
                        if random.random() < 0.5 or len(cur_A) <= len(cur_B):
                            cur_A = cand_A
                            dA.update(new_dA)
                            continue
                            
                # Try adding to B
                cand_B = cur_B + [x]
                if _is_sidon(cand_B):
                    new_dB = set(abs(x - b) for b in cur_B)
                    if not (new_dB & dA):
                        cur_B = cand_B
                        dB.update(new_dB)
                        continue
                        
            size = len(cur_A) + len(cur_B)
            if size > best_seed_size:
                best_seed_size = size
                best_seed = (cur_A, cur_B)
                
        return best_seed

    A, B = _greedy_seed()
    best_score = _score_sets(A, B)
    
    # Phase 2: Simulated Annealing
    # State: A, B
    # Moves:
    # 1. Move element from A to B (if valid)
    # 2. Move element from B to A (if valid)
    # 3. Remove element from A
    # 4. Remove element from B
    # 5. Insert random element into A (if valid)
    # 6. Insert random element into B (if valid)
    # 7. Swap element in A with random candidate
    # 8. Swap element in B with random candidate
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    current_A = list(A)
    current_B = list(B)
    current_score = best_score
    
    # Precompute all possible candidates? No, just sample from 1..256
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        # Generate a move
        move_type = random.choice(['move_AB', 'move_BA', 'remove_A', 'remove_B', 'insert_A', 'insert_B', 'swap_A', 'swap_B'])
        
        new_A = list(current_A)
        new_B = list(current_B)
        valid_move = True
        
        if move_type == 'move_AB' and len(current_A) > 0:
            idx = random.randint(0, len(current_A) - 1)
            x = current_A[idx]
            new_A.pop(idx)
            # Check if x can go to B
            cand_B = new_B + [x]
            if _is_sidon(cand_B):
                # Check diffs
                # We need to recompute diffs for new_A and cand_B
                # This is expensive. 
                # Optimization: Only do this if we are going to evaluate score.
                # SA requires fast evaluation.
                # Let's rely on _score_sets which does the checks.
                # But we need to ensure we don't do invalid moves that are obviously bad?
                # No, let _score_sets handle it.
                new_B = cand_B
            else:
                valid_move = False
                
        elif move_type == 'move_BA' and len(current_B) > 0:
            idx = random.randint(0, len(current_B) - 1)
            x = current_B[idx]
            new_B.pop(idx)
            cand_A = new_A + [x]
            if _is_sidon(cand_A):
                new_A = cand_A
            else:
                valid_move = False
                
        elif move_type == 'remove_A' and len(current_A) > 0:
            idx = random.randint(0, len(current_A) - 1)
            new_A.pop(idx)
            
        elif move_type == 'remove_B' and len(current_B) > 0:
            idx = random.randint(0, len(current_B) - 1)
            new_B.pop(idx)
            
        elif move_type == 'insert_A':
            x = random.randint(1, 256)
            if x not in new_A:
                new_A.append(x)
            else:
                valid_move = False
                
        elif move_type == 'insert_B':
            x = random.randint(1, 256)
            if x not in new_B:
                new_B.append(x)
            else:
                valid_move = False
                
        elif move_type == 'swap_A' and len(current_A) > 0:
            idx = random.randint(0, len(current_A) - 1)
            old_x = current_A[idx]
            new_x = random.randint(1, 256)
            if new_x not in new_A:
                new_A[idx] = new_x
            else:
                valid_move = False
                
        elif move_type == 'swap_B' and len(current_B) > 0:
            idx = random.randint(0, len(current_B) - 1)
            old_x = current_B[idx]
            new_x = random.randint(1, 256)
            if new_x not in new_B:
                new_B[idx] = new_x
            else:
                valid_move = False
        
        if not valid_move:
            continue
            
        new_score = _score_sets(new_A, new_B)
        
        # Acceptance criterion
        if new_score > current_score:
            current_A = new_A
            current_B = new_B
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                A = list(current_A)
                B = list(current_B)
        else:
            # Accept with probability exp((new - cur) / T)
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if random.random() < prob:
                    current_A = new_A
                    current_B = new_B
                    current_score = new_score
            
        T *= cooling_rate
        
    # Final cleanup: ensure we return the best valid sets found
    # If best_score is negative, it means we never found a fully valid set?
    # _score_sets returns negative for invalid.
    # We should return the best valid one.
    # If A and B are empty, we need non-empty.
    if not A or not B:
        # Fallback to simple valid sets
        A = [1]
        B = [2]
        
    return [A, B]

# EVOLVE-BLOCK-END
