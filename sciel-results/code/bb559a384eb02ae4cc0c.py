# EVOLVE-BLOCK-START
import time
import random
import sys

def _check_sidon(S):
    """Check Sidon property: all pair sums (with replacement) are unique."""
    seen = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _diffs(S):
    """Return set of absolute nonzero differences."""
    d = set()
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(S[i] - S[j])
            if diff > 0:
                d.add(diff)
    return d

def _validate(A, B, max_n=256):
    """Validate solution. Returns (is_valid, score)."""
    if not A or not B:
        return False, 0
    if len(A) > 64 or len(B) > 64:
        return False, 0
    for x in A:
        if x < 1 or x > max_n:
            return False, 0
    for x in B:
        if x < 1 or x > max_n:
            return False, 0
    if not _check_sidon(A):
        return False, 0
    if not _check_sidon(B):
        return False, 0
    dA = _diffs(A)
    dB = _diffs(B)
    if dA & dB:
        return False, 0
    return True, len(A) + len(B)

def construct():
    """
    Main entry point. Uses Simulated Annealing to maximize |A|+|B|.
    State: (A, B) where A, B are lists of integers.
    Moves: Add, Remove, Swap (between A and B or within).
    Objective: Maximize size, penalize violations.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    max_n = 256
    max_size = 64

    best_A, best_B = [1], [3]
    best_score = 2
    best_valid = True

    def calc_penalty(A, B):
        """
        Calculate penalty for invalid state.
        Returns (penalty, size_score).
        Lower penalty is better. If penalty is 0, state is valid.
        """
        penalty = 0
        
        # Check sizes
        if len(A) > max_size:
            penalty += (len(A) - max_size) * 10
        if len(B) > max_size:
            penalty += (len(B) - max_size) * 10
            
        if not A or not B:
            return 1000, 0
            
        # Check Sidon A
        seen_sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in seen_sums_A:
                    penalty += 10
                seen_sums_A.add(s)
                
        # Check Sidon B
        seen_sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in seen_sums_B:
                    penalty += 10
                seen_sums_B.add(s)
                
        # Check Diff Disjointness
        dA = _diffs(A)
        dB = _diffs(B)
        common = dA & dB
        if common:
            penalty += len(common) * 5
            
        size_score = len(A) + len(B)
        return penalty, size_score

    # Initialize with a decent greedy solution
    def greedy_init(rng):
        A = []
        B = []
        used_diffs_A = set()
        used_diffs_B = set()
        
        # Random partition of differences
        pool_A = set()
        pool_B = set()
        for d in range(1, max_n):
            r = rng.random()
            if r < 0.45:
                pool_A.add(d)
            elif r < 0.90:
                pool_B.add(d)
        
        if not pool_A: pool_A.add(1)
        if not pool_B: pool_B.add(2)
        
        sums_A = set()
        sums_B = set()
        
        for x in range(1, max_n + 1):
            # Try A
            if len(A) < max_size:
                ok = True
                for a in A:
                    if x + a in sums_A or 2*x in sums_A:
                        ok = False; break
                    d = abs(x-a)
                    if d > 0 and (d not in pool_A or d in used_diffs_B):
                        ok = False; break
                if ok:
                    A.append(x)
                    for a in A[:-1]:
                        sums_A.add(x+a)
                    sums_A.add(2*x)
                    for a in A[:-1]:
                        d = abs(x-a)
                        if d > 0: used_diffs_A.add(d)
                    continue
            
            # Try B
            if len(B) < max_size:
                ok = True
                for b in B:
                    if x + b in sums_B or 2*x in sums_B:
                        ok = False; break
                    d = abs(x-b)
                    if d > 0 and (d not in pool_B or d in used_diffs_A):
                        ok = False; break
                if ok:
                    B.append(x)
                    for b in B[:-1]:
                        sums_B.add(x+b)
                    sums_B.add(2*x)
                    for b in B[:-1]:
                        d = abs(x-b)
                        if d > 0: used_diffs_B.add(d)
                    continue
                    
        return A, B

    rng = random.Random(42)
    
    # Run multiple restarts of SA
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize
        A, B = greedy_init(rng)
        if not A: A = [1]
        if not B: B = [2]
        
        current_A = list(A)
        current_B = list(B)
        current_penalty, current_size = calc_penalty(current_A, current_B)
        
        # Update best
        if current_penalty == 0 and current_size > best_score:
            best_score = current_size
            best_A = list(current_A)
            best_B = list(current_B)
            
        # SA Parameters
        T_start = 100.0
        T_end = 0.1
        steps = 5000
        alpha = (T_end / T_start) ** (1.0 / steps)
        
        T = T_start
        
        for step in range(steps):
            if time.time() > deadline - 2.0:
                break
                
            # Generate Move
            move_type = rng.randint(0, 3)
            
            new_A = list(current_A)
            new_B = list(current_B)
            
            if move_type == 0: # Add to A
                if len(new_A) < max_size:
                    x = rng.randint(1, max_n)
                    if x not in new_A:
                        new_A.append(x)
            elif move_type == 1: # Add to B
                if len(new_B) < max_size:
                    x = rng.randint(1, max_n)
                    if x not in new_B:
                        new_B.append(x)
            elif move_type == 2: # Remove from A or B
                if rng.random() < 0.5 and len(new_A) > 1:
                    idx = rng.randint(0, len(new_A)-1)
                    new_A.pop(idx)
                elif len(new_B) > 1:
                    idx = rng.randint(0, len(new_B)-1)
                    new_B.pop(idx)
            elif move_type == 3: # Swap between A and B
                if new_A and new_B:
                    idx_a = rng.randint(0, len(new_A)-1)
                    idx_b = rng.randint(0, len(new_B)-1)
                    new_A[idx_a], new_B[idx_b] = new_B[idx_b], new_A[idx_a]
            
            # Evaluate
            new_penalty, new_size = calc_penalty(new_A, new_B)
            
            # Objective: Minimize penalty, Maximize size
            # Combine into single score: Score = -size * 10 + penalty
            # Or just use penalty primarily, size as tie-breaker?
            # Let's use: Energy = penalty - size * 2
            # If penalty is 0, we want max size.
            # If penalty > 0, we want min penalty.
            
            old_energy = current_penalty - current_size * 2
            new_energy = new_penalty - new_size * 2
            
            delta_e = new_energy - old_energy
            
            if delta_e < 0:
                # Better
                current_A, current_B = new_A, new_B
                current_penalty, current_size = new_penalty, new_size
                if current_penalty == 0 and current_size > best_score:
                    best_score = current_size
                    best_A = list(current_A)
                    best_B = list(current_B)
            else:
                # Worse: accept with probability exp(-delta_e / T)
                if T > 0:
                    prob = 2.71828 ** (-delta_e / T)
                    if rng.random() < prob:
                        current_A, current_B = new_A, new_B
                        current_penalty, current_size = new_penalty, new_size
                        if current_penalty == 0 and current_size > best_score:
                            best_score = current_size
                            best_A = list(current_A)
                            best_B = list(current_B)
            
            T *= alpha
            if T < T_end:
                T = T_end

    # Final validation
    valid, score = _validate(best_A, best_B)
    if not valid:
        # Fallback to a known valid small set
        return [[1], [3]]
    
    return [best_A, best_B]

# EVOLVE-BLOCK-END
