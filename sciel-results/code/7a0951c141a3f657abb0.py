"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 256
    max_size = 64
    
    def is_sidon(S):
        """Check if set S is a Sidon set (unique pair sums)."""
        S = list(S)
        m = len(S)
        if m == 0:
            return True
        # Using a set of sums is O(m^2). For m<=64, this is ~4096 ops.
        # We need to do this fast.
        sums = set()
        for i in range(m):
            ai = S[i]
            for j in range(i, m):
                s = ai + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_diffs(S):
        """Get set of non-zero differences for S."""
        S = list(S)
        m = len(S)
        diffs = set()
        for i in range(m):
            ai = S[i]
            for j in range(m):
                if i != j:
                    d = S[j] - ai
                    if d != 0:
                        diffs.add(d)
        return diffs

    def get_sums_set(S):
        """Get set of pair sums for S."""
        S = list(S)
        m = len(S)
        sums = set()
        for i in range(m):
            ai = S[i]
            for j in range(i, m):
                sums.add(ai + S[j])
        return sums

    # Initial state: Random valid Sidon sets with disjoint diffs
    # We will try to build a good starting point using a simple greedy method first,
    # then improve with SA.
    
    def greedy_initial():
        A = set()
        B = set()
        diffs_A = set()
        diffs_B = set()
        sums_A = set()
        sums_B = set()
        
        # Sort candidates randomly to avoid bias
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Try adding to A
            if len(A) < max_size:
                # Check Sidon for A
                new_sums_A = sums_A | {x + a for a in A} | {2*x}
                if len(new_sums_A) == len(sums_A) + len(A) + 1:
                    # Check Diffs
                    new_diffs_A = diffs_A | {x - a for a in A} | {a - x for a in A}
                    if not (new_diffs_A & diffs_B):
                        A.add(x)
                        diffs_A = new_diffs_A
                        sums_A = new_sums_A
                        continue
            
            # Try adding to B
            if len(B) < max_size:
                new_sums_B = sums_B | {x + b for b in B} | {2*x}
                if len(new_sums_B) == len(sums_B) + len(B) + 1:
                    new_diffs_B = diffs_B | {x - b for b in B} | {b - x for b in B}
                    if not (new_diffs_B & diffs_A):
                        B.add(x)
                        diffs_B = new_diffs_B
                        sums_B = new_sums_B
                        continue
                        
        return A, B

    A, B = greedy_initial()
    
    # If greedy fails to produce non-empty sets, fallback to simple valid sets
    if not A or not B:
        A = {1, 2, 4}
        B = {8, 16, 32}
        # Verify these are valid
        # A={1,2,4}: sums 2,3,4,5,6,8. diffs: 1,2,3, -1,-2,-3.
        # B={8,16,32}: sums 16,24,40,32,48,64. diffs: 8,16,24, -8,-16,-24.
        # Disjoint diffs? Yes.
        
    best_A = set(A)
    best_B = set(B)
    best_score = len(A) + len(B)
    
    # Precompute some structures for faster checks? 
    # Since we are doing swaps, we need to recalculate diffs/sums for the modified set.
    # m is small (<=64), so O(m^2) per check is acceptable if we don't do too many.
    # 60s budget. ~100k-500k iterations possible?
    
    def evaluate(A, B):
        # Returns score (size) if valid, -1 if invalid
        if not is_sidon(A) or not is_sidon(B):
            return -1
        diffs_A = get_diffs(A)
        diffs_B = get_diffs(B)
        if diffs_A & diffs_B:
            return -1
        return len(A) + len(B)

    # Simulated Annealing
    current_A = set(A)
    current_B = set(B)
    current_score = evaluate(current_A, current_B)
    if current_score == -1:
        current_score = 0 # Fallback, should not happen if greedy works
    
    T = 10.0
    cooling = 0.999
    
    # Candidate pool: All numbers 1..256
    all_numbers = list(range(1, n + 1))
    
    while time.time() < deadline:
        # Generate move
        # Types:
        # 1. Add x to A
        # 2. Add x to B
        # 3. Remove x from A
        # 4. Remove x from B
        # 5. Swap x in A with y in B (or vice versa) - complex, skip for now?
        #    Actually, simple add/remove is sufficient to explore.
        
        move_type = random.random()
        
        new_A = set(current_A)
        new_B = set(current_B)
        
        if move_type < 0.25:
            # Add to A
            if len(new_A) < max_size:
                # Pick random candidate not in A
                candidates = [x for x in all_numbers if x not in new_A]
                if candidates:
                    x = random.choice(candidates)
                    new_A.add(x)
        elif move_type < 0.5:
            # Add to B
            if len(new_B) < max_size:
                candidates = [x for x in all_numbers if x not in new_B]
                if candidates:
                    x = random.choice(candidates)
                    new_B.add(x)
        elif move_type < 0.75:
            # Remove from A
            if new_A:
                x = random.choice(list(new_A))
                new_A.remove(x)
        else:
            # Remove from B
            if new_B:
                x = random.choice(list(new_B))
                new_B.remove(x)
        
        # Evaluate new state
        new_score = evaluate(new_A, new_B)
        
        if new_score == -1:
            # Invalid state. Treat as score 0? Or very low?
            # In SA, we need a numerical score. Let's say invalid = 0.
            # But we want to maximize size. 
            # If current is valid (score > 0) and new is invalid (score 0), delta is negative.
            # If current is invalid (score 0) and new is valid, delta is positive.
            # We should prefer valid states.
            # Let's use a penalty. 
            # Actually, the task requires valid sets. 
            # If we accept invalid states, we might get stuck in invalid regions.
            # Better to only accept valid states if current is valid?
            # No, standard SA accepts worse states with probability exp(delta/T).
            # If we map invalid to 0, and valid to size, then moving from valid(60) to invalid(0) is bad.
            # Moving from invalid(0) to valid(10) is good.
            
            if current_score > 0:
                 # We are in a valid state, moving to invalid is a big drop.
                 # Let's just reject invalid moves if we are currently valid, 
                 # or accept with very low probability.
                 # Simpler: If current is valid, only accept valid moves or 
                 # accept invalid with prob exp(-large/T).
                 # Let's just reject invalid moves if current is valid to keep search in valid space.
                 # This turns it into a local search with random restarts? 
                 # No, if we only accept valid moves, we can't escape local minima 
                 # unless we allow temporary invalidity.
                 
                 # Let's try: If new is invalid, reject.
                 continue
            
            # If current is invalid, we accept any valid move, or invalid move with prob?
            # Let's just accept valid moves.
            
        # Both valid or both invalid (score 0)
        delta = new_score - current_score
        
        if delta >= 0:
            current_A = new_A
            current_B = new_B
            current_score = new_score
            
            if current_score > best_score:
                best_A = set(current_A)
                best_B = set(current_B)
                best_score = current_score
        else:
            # Accept with probability exp(delta / T)
            # delta is negative
            p = np.exp(delta / T)
            if random.random() < p:
                current_A = new_A
                current_B = new_B
                current_score = new_score
        
        T *= cooling
        
    # Final check
    if evaluate(best_A, best_B) != len(best_A) + len(best_B):
        # Fallback to a known valid small set
        best_A = {1, 2, 4}
        best_B = {8, 16, 32}
        
    return [sorted(best_A), sorted(best_B)]

# EVOLVE-BLOCK-END
