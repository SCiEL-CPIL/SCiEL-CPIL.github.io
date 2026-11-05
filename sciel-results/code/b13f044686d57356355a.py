"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    MOD = 1009
    N = 1008
    
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    # Precompute modular inverses: inv[x] = x^(-1) mod 1009 for x in [1, 1008]
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Self-inverse residues: x such that x*x % MOD == 1
    # These are x = 1 and x = MOD-1 = 1008
    self_inverse = {1, 1008}
    
    best_start = 2
    best_len = 1
    
    # Two-pointer sliding window approach:
    # For each left pointer S, extend right pointer E as far as possible
    # such that the interval [S, E] is valid.
    # 
    # An interval [S, E] is valid iff:
    #   1. No element in [S, E] is a self-inverse (1 or 1008)
    #   2. For no x, y in [S, E] do we have x*y % MOD == 1
    #      i.e., for no x in [S, E] is inv[x] also in [S, E]
    #
    # Key insight: for a fixed S, as E increases, once a conflict appears,
    # it persists. So we can use a two-pointer where we track which elements
    # in the current window have their inverse also in the window.
    #
    # Efficient approach: for each S, we need the minimum E such that
    # either E is a self-inverse, or there exists x in [S, E] with inv[x] in [S, E].
    #
    # For a given S, the earliest conflict from the right is determined by:
    #   - If S itself is self-inverse: invalid immediately
    #   - For each x in [S, ...], if inv[x] >= S and inv[x] > x, then including
    #     both x and inv[x] creates a conflict. The smallest such inv[x] for
    #     x >= S gives us a bound.
    #   - Also, if 1008 is in range, it's self-inverse.
    #
    # Let's use a different approach: for each possible start S (avoiding self-inverses),
    # find the maximum E such that [S,E] is valid.
    #
    # For a fixed S (not self-inverse), the interval [S, E] is invalid if:
    #   a) E >= 1008 (since 1008 is self-inverse and would be included) -- wait, 1008 is in [1,1008]
    #      Actually the interval is contained in [1, 1008], so E <= 1008.
    #      If S <= 1008 <= E, then 1008 is in the interval and 1008*1008 % 1009 = 1, invalid.
    #   b) There exists x in [S, E] such that inv[x] is also in [S, E].
    #
    # For condition (b): for each x in [S, E], we need inv[x] < S or inv[x] > E.
    # Equivalently, for all x in [S, E]: inv[x] not in [S, E].
    #
    # Let's precompute for each position x, what is the "conflict range" it creates.
    # If x and inv[x] are both in the interval, it's a conflict.
    # For a pair (x, inv[x]) with x != inv[x], the interval [S, E] contains both
    # iff S <= min(x, inv[x]) and max(x, inv[x]) <= E.
    #
    # So for a fixed S, the maximum valid E is:
    #   E_max = min over all pairs (a, b) where a = min(x, inv[x]), b = max(x, inv[x])
    #            with a >= S, of (a - 1)  [i.e., we can't reach b if a >= S]
    #   Also E_max <= 1007 (to avoid including 1008)
    #   Also we need S != 1 (to avoid including 1)
    #
    # Wait, let me reconsider. For a fixed S:
    #   - S must not be 1 or 1008 (self-inverse)
    #   - E must not reach 1008 if S <= 1008 (always true since S >= 2), so E <= 1007
    #   - For every pair (x, inv[x]) where x != inv[x]: if S <= min(x,inv[x]) and max(x,inv[x]) <= E, conflict.
    #     So E must be < max(x, inv[x]) for all x where min(x, inv[x]) >= S.
    #     Equivalently: E < min{ max(x, inv[x]) : x in [S, 1008], x != inv[x], min(x,inv[x]) >= S }
    #     But wait, min(x, inv[x]) >= S is automatically satisfied if x >= S and inv[x] >= S.
    #     Actually the condition is: both x and inv[x] are in [S, E].
    #     This means S <= x <= E and S <= inv[x] <= E.
    #     So we need: for all x, NOT (S <= x <= E AND S <= inv[x] <= E).
    #
    # For a fixed S, define:
    #   conflict_E = min{ max(x, inv[x]) : x in [S, 1008] such that inv[x] >= S and x != inv[x] }
    #   Then E can go up to conflict_E - 1 (if conflict_E exists), or up to 1007 otherwise.
    #   Also E must be >= S.
    #
    # But we also need to check: is it possible that x < S but inv[x] >= S? Then x is not in [S,E]
    # so no conflict from that x. Similarly if inv[x] < S, no conflict.
    # The conflict only arises when BOTH x and inv[x] are >= S and <= E.
    #
    # So for fixed S:
    #   E_max = 1007  (can't include 1008)
    #   For each x in [S, 1008]:
    #     if inv[x] >= S and x != inv[x]:
    #       # Both x and inv[x] are >= S. If E >= max(x, inv[x]), conflict.
    #       E_max = min(E_max, max(x, inv[x]) - 1)
    #   Also need E_max >= S for valid interval.
    #   If S == 1 or S == 1008: skip.
    
    # Precompute: for each S, what is the minimum max(x, inv[x]) over all x >= S with inv[x] >= S?
    # This is a bit complex. Let's just do it directly for each S.
    # N = 1008, so O(N^2) = ~1 million operations, very fast.
    
    for S in range(2, MOD):  # S from 2 to 1007 (avoid 1 and 1008)
        if time.time() > deadline:
            break
        
        if S == 1008:
            continue
            
        E_max = 1007  # Can't include 1008
        
        # Check all x in [S, 1007] (since 1008 is excluded from valid intervals when S <= 1008)
        # Actually x can be up to 1007 since E <= 1007
        for x in range(S, 1008):
            ix = inv[x]
            if ix == x:
                # Self-inverse within range, but we already excluded 1 and 1008
                # In mod 1009, only 1 and 1008 are self-inverse, so this won't happen for x in [2,1007]
                continue
            if ix >= S:
                # Both x and ix are >= S
                # If E >= max(x, ix), conflict
                m = max(x, ix)
                if m - 1 < E_max:
                    E_max = m - 1
                    if E_max < S:
                        break
        
        if E_max >= S:
            current_len = E_max - S + 1
            if current_len > best_len:
                best_len = current_len
                best_start = S
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
