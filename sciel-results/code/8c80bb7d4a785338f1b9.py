"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse residues: x * x % p == 1
    # These are x = 1 and x = p-1 = 1008
    self_inv = set()
    for x in range(1, p):
        if (x * x) % p == 1:
            self_inv.add(x)
    
    # For each x in [1, 1008], the "forbidden partner" is inv[x]
    # An interval [s, e] is valid iff for every x in [s, e]:
    #   1. x is not self-inverse (i.e., x not in self_inv)
    #   2. inv[x] is not in [s, e]
    #
    # Equivalently: for every x in [s, e], inv[x] must be outside [s, e].
    #
    # We want to find the maximum length interval [s, e] with 1 <= s <= e <= 1008.
    #
    # Strategy: For each possible start s, find the maximum valid end e.
    # As we extend e from s upward, the constraint is violated when:
    #   - e itself is self-inverse, OR
    #   - there exists some x in [s, e] such that inv[x] is in [s, e]
    #
    # More precisely, the interval [s, e] is valid iff:
    #   For all x in [s, e]: inv[x] < s or inv[x] > e
    #
    # So for a fixed s, the maximum valid e is the minimum over x in [s, e] of 
    # (the point where inv[x] enters the interval). But this is circular.
    #
    # Better approach: For each x, define the "constraint interval" that x forbids.
    # x forbids any interval [s, e] such that s <= x <= e AND s <= inv[x] <= e.
    # i.e., the interval must contain both x and inv[x].
    #
    # So an interval [s, e] is valid iff for NO x does it contain both x and inv[x].
    #
    # For a fixed s, what is the maximum e such that no x in [s, e] has inv[x] in [s, e]?
    # This means: for all x in [s, e], inv[x] < s or inv[x] > e.
    #
    # Let's think of it differently. For each x, the pair {x, inv[x]} must not both be in [s,e].
    # (Note: if x = inv[x], i.e., self-inverse, then x alone in [s,e] already violates because x*x%p=1.)
    #
    # So the condition is: [s,e] does not contain both elements of any pair {x, inv[x]},
    # and [s,e] does not contain any self-inverse element.
    #
    # For a fixed s, the maximum valid e:
    # Start with e = s. Check if s is self-inverse -> if so, invalid.
    # Then extend e one by one. At each step, check if the new e creates a violation.
    # A violation occurs if:
    #   - e is self-inverse, OR
    #   - inv[e] is in [s, e-1] (since inv[e] could be in the existing part)
    #   - OR there exists x in [s, e-1] such that inv[x] == e (but that's the same as inv[e] == x)
    #
    # Actually, when we add e to the interval, the only new pairs to check are those involving e.
    # So we need: inv[e] is not in [s, e-1] (and e is not self-inverse).
    #
    # But wait, we also need to ensure that for all existing x in [s, e-1], inv[x] is not e.
    # But inv[x] = e iff x = inv[e]. So we just need inv[e] not in [s, e-1].
    #
    # So the algorithm for fixed s:
    #   If s is self-inverse, skip.
    #   e = s
    #   For candidate_e from s+1 to 1008:
    #       If candidate_e is self-inverse: break
    #       If inv[candidate_e] >= s and inv[candidate_e] <= candidate_e - 1: break
    #       Actually, inv[candidate_e] must not be in [s, candidate_e].
    #       But inv[candidate_e] != candidate_e (since not self-inverse), so we check:
    #       If s <= inv[candidate_e] < candidate_e: break (inv is in the existing part)
    #       If inv[candidate_e] == candidate_e: already handled by self-inverse check
    #       If inv[candidate_e] > candidate_e: that's fine for now, but when we reach inv[candidate_e], we'll check.
    #
    # Wait, this isn't quite right. When we add candidate_e, we need inv[candidate_e] not in [s, candidate_e].
    # Since inv[candidate_e] != candidate_e (not self-inverse), we need:
    #   inv[candidate_e] < s OR inv[candidate_e] > candidate_e
    #
    # But also, for all existing x in [s, candidate_e - 1], we need inv[x] not in [s, candidate_e].
    # Previously, inv[x] was not in [s, candidate_e - 1]. Now it could be candidate_e.
    # inv[x] == candidate_e iff x == inv[candidate_e].
    # So we need inv[candidate_e] not in [s, candidate_e - 1], which is the same condition.
    #
    # So the incremental check is: for new element e, check if inv[e] is in [s, e-1].
    # If yes, break. Otherwise, continue.
    #
    # Let's implement this.
    
    best_start = 2
    best_len = 0
    
    deadline = time.time() + 80  # 80 second deadline
    
    for s in range(1, n + 1):
        if time.time() > deadline:
            break
        if s in self_inv:
            continue
        
        e = s
        valid = True
        for candidate_e in range(s + 1, n + 1):
            if time.time() > deadline:
                break
            if candidate_e in self_inv:
                valid = False
                break
            ie = inv[candidate_e]
            # Check if ie is in [s, candidate_e - 1]
            if s <= ie < candidate_e:
                valid = False
                break
            e = candidate_e
        
        length = e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
