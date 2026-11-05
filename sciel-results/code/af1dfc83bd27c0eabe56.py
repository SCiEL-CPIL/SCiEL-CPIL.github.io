"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = 256
    # Start with a deterministic pattern that has good low-discrepancy properties
    # Use a Thue-Morse-like sequence or start with alternating signs
    signs = np.ones(n, dtype=np.int8)
    
    # Initialize with a pattern that tends to have low discrepancy
    # Use a simple heuristic: alternate in blocks
    for i in range(n):
        # Create a pattern based on binary representation
        # This gives a more structured starting point than random
        signs[i] = 1 if bin(i).count('1') % 2 == 0 else -1
    
    # Local search: iteratively flip signs to reduce max discrepancy
    # We'll evaluate the maximum absolute sum over all arithmetic progressions
    # and try to flip the sign that most reduces this maximum
    
    def max_discrepancy(s):
        """Compute maximum absolute signed sum over all arithmetic progressions."""
        max_abs = 0
        # Check all possible arithmetic progressions
        # For efficiency, we limit the step and length
        for start in range(n):
            for step in range(1, n - start):
                # Collect elements in this AP
                idx = start
                while idx < n:
                    # We can't easily compute all APs efficiently, so use sampling
                    break
                idx = start
                total = 0
                while idx < n:
                    total += s[idx]
                    max_abs = max(max_abs, abs(total))
                    idx += step
        return max_abs
    
    # Better approach: use a limited set of critical APs for speed
    # Focus on short APs and common steps which tend to give worst cases
    
    def compute_critical_sums(s, n):
        """Compute sums for a set of critical arithmetic progressions."""
        sums = {}
        # Include all singletons (trivially |s[i]| = 1)
        # Include pairs, triples, and short APs with small steps
        for start in range(n):
            for step in range(1, min(20, n - start)):
                # Limit length to keep computation fast
                max_len = min(30, (n - 1 - start) // step + 1)
                for length in range(1, max_len + 1):
                    end = start + (length - 1) * step
                    if end < n:
                        key = (start, step, length)
                        total = 0
                        for k in range(length):
                            total += s[start + k * step]
                        sums[key] = total
        return sums
    
    # Greedy local search
    best_s = signs.copy()
    best_max = 10**9
    
    for iteration in range(50):
        # Compute current max discrepancy on critical APs
        sums = compute_critical_sums(best_s, n)
        current_max = max(abs(v) for v in sums.values())
        
        if current_max < best_max:
            best_max = current_max
        
        # Try flipping each position and see if it reduces the max
        improved = False
        for i in range(n):
            old_val = best_s[i]
            best_s[i] = -old_val
            
            # Recompute affected sums
            # Only APs containing position i are affected
            new_sums = compute_critical_sums(best_s, n)
            new_max = max(abs(v) for v in new_sums.values())
            
            if new_max < current_max:
                improved = True
                current_max = new_max
                break
            else:
                best_s[i] = old_val
        
        if not improved:
            break
    
    return list(best_s)
# EVOLVE-BLOCK-END

