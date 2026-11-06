"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Use simulated annealing on the set of selected indices to find a strictly
    increasing sequence of integers in [1,256] such that all nonempty consecutive
    block sums are distinct. Maximize sequence length.
    """
    
    def is_valid(seq):
        """Check if sequence is strictly increasing and all consecutive block sums are distinct."""
        n = len(seq)
        if n == 0:
            return True
        # Check strictly increasing and bounds
        if n > 0:
            if seq[0] < 1 or seq[-1] > 256:
                return False
            for i in range(1, n):
                if seq[i] <= seq[i-1]:
                    return False
        
        # Check consecutive block sums are distinct
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def greedy_construct():
        """Construct a valid sequence using the smallest possible values."""
        seq = []
        current_sums = set()
        for x in range(1, 257):
            # Compute new sums if we append x
            new_sums = [x]
            s = x
            for a in reversed(seq):
                s += a
                new_sums.append(s)
            
            if not current_sums.intersection(new_sums):
                seq.append(x)
                current_sums.update(new_sums)
        return seq

    def simulated_annealing(initial_seq, time_limit=50.0):
        """Simulated annealing to maximize sequence length while maintaining validity."""
        start_time = time.time()
        current = list(initial_seq)
        current_len = len(current)
        best = list(current)
        best_len = current_len
        
        temp = 10.0
        cooling = 0.999
        
        all_values = list(range(1, 257))
        
        while time.time() - start_time < time_limit:
            temp = max(0.01, temp * cooling)
            
            move_type = random.random()
            new_seq = None
            
            if move_type < 0.5:
                # Try to add an element
                if len(current) < 20: # Heuristic cap to avoid huge O(N^2) costs, though N is small
                    pos = random.randint(0, len(current))
                    low = current[pos-1] + 1 if pos > 0 else 1
                    high = current[pos] - 1 if pos < len(current) else 256
                    if low <= high:
                        # Try a few random values in the gap
                        for _ in range(3):
                            new_val = random.randint(low, high)
                            new_seq = current[:pos] + [new_val] + current[pos:]
                            if is_valid(new_seq):
                                break
                        if new_seq is None or not is_valid(new_seq):
                            new_seq = None
            elif move_type < 0.8:
                # Try to replace an element
                if len(current) > 0:
                    pos = random.randint(0, len(current) - 1)
                    low = current[pos-1] + 1 if pos > 0 else 1
                    high = current[pos+1] - 1 if pos < len(current) - 1 else 256
                    if low <= high:
                        new_val = random.randint(low, high)
                        if new_val != current[pos]:
                            new_seq = current[:pos] + [new_val] + current[pos+1:]
                            if not is_valid(new_seq):
                                new_seq = None
            else:
                # Try to remove an element
                if len(current) > 1:
                    pos = random.randint(0, len(current) - 1)
                    new_seq = current[:pos] + current[pos+1:]
                    if not is_valid(new_seq):
                        new_seq = None

            if new_seq is None:
                continue

            new_len = len(new_seq)
            delta_len = new_len - current_len
            
            # Acceptance criterion
            if delta_len > 0:
                # Always accept improvement
                current = new_seq
                current_len = new_len
                if new_len > best_len:
                    best = new_seq
                    best_len = new_len
            elif delta_len == 0:
                # Accept sideways moves with some probability to explore
                if random.random() < 0.1:
                    current = new_seq
            else:
                # Accept worse moves with probability exp(delta_len / temp)
                prob = np.exp(delta_len / temp)
                if random.random() < prob:
                    current = new_seq
                    current_len = new_len
        
        return best

    # 1. Get a strong baseline using greedy
    greedy_seq = greedy_construct()
    best_overall = greedy_seq
    best_overall_len = len(greedy_seq)
    
    # 2. Run SA starting from the greedy solution
    # We run SA for a few seconds to try to improve upon greedy
    # Greedy is deterministic, so we use the same start but random moves in SA
    sa_result = simulated_annealing(greedy_seq, time_limit=45.0)
    
    if len(sa_result) > best_overall_len:
        best_overall = sa_result
        best_overall_len = len(sa_result)
        
    # 3. Try a few random restarts with shorter time limits if time permits
    # (Optional, but SA from greedy is usually strong)
    
    return best_overall
# EVOLVE-BLOCK-END
