"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    N = 1000
    chosen = []
    counts = {}
    
    def can_add(x):
        """Check if adding x keeps all sums <= 2."""
        for y in chosen:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        if counts.get(2 * x, 0) >= 2:
            return False
        return True
    
    def add_element(x):
        """Add x to chosen and update counts."""
        for y in chosen:
            s = x + y
            counts[s] = counts.get(s, 0) + 1
        s2 = 2 * x
        counts[s2] = counts.get(s2, 0) + 1
        chosen.append(x)
    
    def remove_element(x):
        """Remove x from chosen and update counts."""
        for y in chosen:
            if y == x:
                continue
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        s2 = 2 * x
        counts[s2] -= 1
        if counts[s2] == 0:
            del counts[s2]
        chosen.remove(x)
    
    # Greedy with lookahead: prefer elements that create fewer "tight" sums
    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        
        if can_add(x):
            # Look ahead: check how many future elements would be blocked
            # by adding x. Prefer x that blocks fewer future elements.
            # Simple heuristic: count how many sums would reach exactly 2
            tight_count = 0
            for y in chosen:
                s = x + y
                if counts.get(s, 0) == 1:
                    tight_count += 1
            if counts.get(2 * x, 0) == 1:
                tight_count += 1
            
            # If this creates many tight sums, consider skipping
            # But for now, just add it (simple greedy)
            add_element(x)
    
    return chosen
# EVOLVE-BLOCK-END
