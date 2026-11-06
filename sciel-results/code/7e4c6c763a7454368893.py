<<<<<<< SEARCH
"""Rewritten solver: precomputed constraints + vectorized greedy + local-search repair."""
import numpy as np
import time

def construct():
    t0 = time.time()
    N = 128
    # positions 1..128, index 0..127
    # A 4-AP: (a, a+d, a+2d, a+3d) with d>=1, a+3d <= 128
    # For each position p (1-based), and role r in 0..3, list of (d, other_positions)
    # Actually we need: for position p, for each AP it's in, the other 3 positions.
    # Precompute: for each p (0-based), for each role r, list of AP indices where p is the r-th term.
    
    # Simpler: precompute for each p, a list of (other1, other2, other3) for each AP containing p.
    # But that's a lot. Instead, precompute the constraint as: for each p, for each AP, the set of other positions.
    
    # Let's precompute: ap_list[p] = list of tuples (o1, o2, o3) where p is part of a 4-AP with those other 3.
    # For greedy, when placing p, we need to check all APs containing p: the colors of the 3 others must not all be same as p's color AND not all 4 same.
    # Actually the constraint is: at least 3 distinct colors among the 4.
    # Violation: fewer than 3 distinct colors, i.e., 1 or 2 distinct colors.
    # 1 distinct: all 4 same.
    # 2 distinct: 3 same + 1 different, or 2+2.
    
    # For greedy at position p (0-based), given colors[0..p-1] are set:
    # For each AP containing p: the other 3 positions must all be < p for the constraint to be fully checkable.
    # But some APs containing p may have positions > p. Those can't be checked yet.
    # So for greedy, we only check APs where p is the LAST position (largest index).
    # That means: APs where p = a+3d for some a, d.
    
    # Precompute: for each p (0-based, representing position p+1), list of APs where p is the 4th term.
    # AP: (a, a+d, a+2d, a+3d), 1-based. p is 4th term: p+1 = a+3d, so a = p+1-3d, a>=1, d>=1.
    # For such AP, the other 3 positions are (a-1, a+d-1, a+2d-1) in 0-based.
    
    last_term_aps = [[] for _ in range(N)]  # last_term_aps[p] = list of (o1, o2, o3) 0-based
    for p in range(N):
        pos = p + 1
        for d in range(1, pos//3 + 1):
            a = pos - 3*d
            if a < 1:
                break
            o1 = a - 1
            o2 = a + d - 1
            o3 = a + 2*d - 1
            last_term_aps[p].append((o1, o2, o3))
    
    # Greedy: place positions 0..127, choosing color to minimize violations in APs where this is last term.
    # We'll use 2 colors first (0,1) and see if it works. If not, try 3.
    
    def greedy(n_colors):
        colors = np.full(N, -1, dtype=np.int8)
        for p in range(N):
            best_color = 0
            best_viol = None
            for c in range(n_colors):
                viol = 0
                for (o1, o2, o3) in last_term_aps[p]:
                    c1, c2, c3 = colors[o1], colors[o2], colors[o3]
                    # Count distinct colors among {c, c1, c2, c3}
                    # Violation if < 3 distinct
                    distinct = len({c, c1, c2, c3})
                    if distinct < 3:
                        viol += 1
                if best_viol is None or viol < best_viol:
                    best_viol = viol
                    best_color = c
            colors[p] = best_color
        return colors
    
    # Try 2 colors
    colors2 = greedy(2)
    # Check if valid
    def count_violations(colors):
        total = 0
        for p in range(N):
            for (o1, o2, o3) in last_term_aps[p]:
                distinct = len({colors[p], colors[o1], colors[o2], colors[o3]})
                if distinct < 3:
                    total += 1
        return total
    
    viol2 = count_violations(colors2)
    
    if viol2 == 0:
        # 2 colors works! Return it.
        # But wait, the problem says "at least three distinct colors", so 2 colors can NEVER satisfy this.
        # Every 4-AP must have at least 3 distinct colors. With only 2 colors total, it's impossible.
        # So 2 colors will always have violations. Let's try 3.
        pass
    
    # Try 3 colors
    colors3 = greedy(3)
    viol3 = count_violations(colors3)
    
    # Local search repair on the best starting point
    best_colors = colors3.copy()
    best_viol = viol3
    
    # Local search: try changing one position at a time
    deadline = t0 + 80  # safety margin
    
    for iteration in range(1000):
        if time.time() > deadline:
            break
        improved = False
        for p in range(N):
            if time.time() > deadline:
                break
            current_color = best_colors[p]
            for c in range(3):
                if c == current_color:
                    continue
                # Try changing p to c
                best_colors[p] = c
                # Recompute violations only for APs where p is last term, plus APs where p is 1st, 2nd, 3rd term
                # This is complex. Let's just recompute all violations (N=128, should be fast enough)
                new_viol = count_violations(best_colors)
                if new_viol < best_viol:
                    best_viol = new_viol
                    improved = True
                    break
                else:
                    best_colors[p] = current_color
            if not improved and best_viol == 0:
                break
        if best_viol == 0:
            break
    
    # If still violations, try 4 colors
    if best_viol > 0:
        colors4 = greedy(4)
        viol4 = count_violations(colors4)
        if viol4 < best_viol:
            best_colors = colors4
            best_viol = viol4
            # Local search with 4 colors
            for iteration in range(500):
                if time.time() > deadline:
                    break
                improved = False
                for p in range(N):
                    if time.time() > deadline:
                        break
                    current_color = best_colors[p]
                    for c in range(4):
                        if c == current_color:
                            continue
                        best_colors[p] = c
                        new_viol = count_violations(best_colors)
                        if new_viol < best_viol:
                            best_viol = new_viol
                            improved = True
                            break
                        else:
                            best_colors[p] = current_color
                    if best_viol == 0:
                        break
                if best_viol == 0:
                    break
    
    # Convert to list of ints
    result = [int(c) for c in best_colors]
    return result
=======
import numpy as np
import time
import random

def construct():
    t0 = time.time()
    N = 128
    deadline = t0 + 55.0  # Safety margin
    
    # Precompute all 4-APs
    # AP: (a, a+d, a+2d, a+3d) 0-based indices
    # a >= 0, a+3d < N
    aps = []
    ap_indices = [[] for _ in range(N)] # ap_indices[p] = list of AP indices containing p
    
    idx = 0
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(0, N - 3 * d):
            p0 = a
            p1 = a + d
            p2 = a + 2 * d
            p3 = a + 3 * d
            aps.append((p0, p1, p2, p3))
            ap_indices[p0].append(idx)
            ap_indices[p1].append(idx)
            ap_indices[p2].append(idx)
            ap_indices[p3].append(idx)
            idx += 1
    
    M = len(aps)
    
    # Helper to check if an AP is violating given colors
    def is_violating(ap, colors):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c0, c1, c2, c3}) < 3

    # Greedy initialization with 3 colors
    # We want to minimize violations.
    # Simple greedy: assign color to p that minimizes new violations in APs where p is the largest index?
    # Or just random and fix?
    # Let's use a greedy that checks all APs containing p.
    
    def greedy_init():
        colors = np.zeros(N, dtype=np.int8)
        for p in range(N):
            best_c = 0
            best_viol = 10**9
            for c in range(3):
                colors[p] = c
                viol = 0
                for ap_idx in ap_indices[p]:
                    if is_violating(aps[ap_idx], colors):
                        viol += 1
                if viol < best_viol:
                    best_viol = viol
                    best_c = c
            colors[p] = best_c
        return colors

    colors = greedy_init()
    
    # Calculate initial violations
    ap_viol = np.zeros(M, dtype=np.int8)
    total_viol = 0
    for i in range(M):
        if is_violating(aps[i], colors):
            ap_viol[i] = 1
            total_viol += 1

    # Local Search
    # We track ap_viol to update quickly.
    # When changing color of p:
    # 1. For each ap in ap_indices[p]:
    #    - If ap_viol[ap] == 1, total_viol -= 1
    #    - Check new violation status
    #    - If new status == 1, total_viol += 1
    #    - Update ap_viol[ap]
    
    best_colors = colors.copy()
    best_total_viol = total_viol
    
    # Simulated Annealing or Hill Climbing?
    # Hill Climbing with random restarts if stuck.
    
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart > 0:
            # Random restart: random 3-coloring
            colors = np.random.randint(0, 3, N).astype(np.int8)
            ap_viol = np.zeros(M, dtype=np.int8)
            total_viol = 0
            for i in range(M):
                if is_violating(aps[i], colors):
                    ap_viol[i] = 1
                    total_viol += 1
            
        if total_viol == 0:
            best_colors = colors.copy()
            best_total_viol = 0
            break
        
        # Hill Climbing
        no_improve = 0
        max_steps = 1000
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if total_viol == 0:
                best_colors = colors.copy()
                best_total_viol = 0
                break
            
            # Pick a random position
            p = np.random.randint(N)
            old_c = colors[p]
            new_c = np.random.randint(3)
            if new_c == old_c:
                continue
            
            # Calculate delta
            delta = 0
            for ap_idx in ap_indices[p]:
                old_v = ap_viol[ap_idx]
                # Temporarily set new color
                colors[p] = new_c
                new_v = 1 if is_violating(aps[ap_idx], colors) else 0
                if new_v != old_v:
                    delta += new_v - old_v
                # Restore old color for now? No, we commit if better.
                # But we need to calculate total delta first.
                # Let's just apply and check total.
                # Actually, calculating delta is safer.
                # Let's revert colors[p] to old_c to keep state consistent during delta calc?
                # No, we can just compute new_v based on new_c.
                # is_violating uses colors array. If we change colors[p], it affects all APs.
                # So we must change colors[p] to new_c, calc new_v, then decide.
                # If we decide to revert, we change back.
                # Let's just do the update directly.
                
                # Optimization: We already set colors[p] = new_c.
                # We need to update ap_viol[ap_idx] and total_viol.
                # But we are still in the loop. We should accumulate delta first.
                # Let's revert colors[p] to old_c immediately after checking?
                # No, is_violating depends on the current state of colors.
                # So:
                # 1. Save old_c
                # 2. Set colors[p] = new_c
                # 3. For each ap, calc new_v.
                # 4. If total delta < 0, accept. Else revert.
                
                # This approach requires reverting if rejected.
                pass # Placeholder logic below

            # Re-implementation of step logic for clarity and speed
            
            # We need to be careful with the state.
            # Let's try a move.
            colors[p] = new_c
            
            # Calculate change in total violations
            delta = 0
            for ap_idx in ap_indices[p]:
                old_v = ap_viol[ap_idx]
                new_v = 1 if is_violating(aps[ap_idx], colors) else 0
                if new_v != old_v:
                    delta += new_v - old_v
            
            if delta < 0:
                # Accept move
                total_viol += delta
                # Update ap_viol
                for ap_idx in ap_indices[p]:
                    ap_viol[ap_idx] = 1 if is_violating(aps[ap_idx], colors) else 0
                no_improve = 0
            else:
                # Reject move, revert
                colors[p] = old_c
                no_improve += 1
                
                if no_improve > 100:
                    break

    # Check if we found a solution
    if best_total_viol > 0:
        # Try 4 colors? Unlikely to be needed if 3 works.
        # But if 3 fails, 4 is the next integer.
        # The problem minimizes labels used. 3 is the target.
        # If 3 is impossible (which it isn't for N=128? Wait, is 3 colors sufficient for N=128?)
        # Van der Waerden number W(3, 4) is 76.
        # This means any 3-coloring of 1..76 contains a monochromatic 4-AP.
        # The constraint is "at least 3 distinct colors".
        # A monochromatic 4-AP has 1 distinct color. Violation.
        # So we must avoid monochromatic 4-APs AND 2-color 4-APs.
        # W(3,4)=76 implies that for N>=76, we cannot avoid monochromatic 4-APs with 3 colors?
        # No, W(3,4)=76 means there EXISTS a 3-coloring of 1..75 with NO monochromatic 4-AP.
        # And for 1..76, EVERY 3-coloring has a monochromatic 4-AP.
        # So for N=128, it is IMPOSSIBLE to have a 3-coloring with no monochromatic 4-AP.
        # Therefore, a 3-coloring MUST have at least one monochromatic 4-AP.
        # Thus, 3 colors are NOT sufficient to satisfy "at least 3 distinct colors" for all 4-APs.
        # We need at least 4 colors?
        
        # Let's check W(4,4). W(4,4) is 76655.
        # So there exists a 4-coloring of 1..76654 with no monochromatic 4-AP.
        # So 4 colors are sufficient to avoid monochromatic 4-APs.
        # But we also need to avoid 2-color 4-APs.
        # It is highly likely that 4 colors are sufficient for N=128.
        
        # So the minimum number of colors is likely 4.
        
        # Let's run the search with 4 colors.
        best_colors = greedy_init_4()
        # ... similar local search ...
        
    return [int(c) for c in best_colors]

def greedy_init_4():
    # This is a stub, need to implement properly inside construct or as nested
    pass
>>>>>>> REPLACE

Wait, the `greedy_init_4` and the logic inside `construct` are messy in the replacement above. I will clean it up. I need to ensure the code is self-contained and correct.

Also, the `is_violating` function call inside the loop is expensive. I should inline it or use a faster check.
`len({c0, c1, c2, c3}) < 3` is equivalent to:
- All 4 same
- 3 same, 1 diff
- 2 same, 2 same
- 2 same, 1 diff, 1 diff (this is 3 distinct, so OK)
Actually, `len(set) < 3` means size is 1 or 2.
Size 1: c0=c1=c2=c3
Size 2: e.g., c0=c1=c2!=c3 OR c0=c1!=c2=c3 OR c0=c2=c3!=c1 etc.

A faster check for "at least 3 distinct":
`c0 != c1 and c0 != c2 and c0 != c3` is NOT enough (c1=c2=c3 is 2 distinct).
We need to ensure that NOT (at most 2 distinct).
At most 2 distinct means:
- c0==c1 and c0==c2 (then c3 can be anything? No, if c0=c1=c2, and c3!=c0, distinct is 2. If c3=c0, distinct is 1. Both bad.)
- So if c0==c1==c2, it's bad regardless of c3.
- If c0==c1==c3, bad.
- If c0==c2==c3, bad.
- If c1==c2==c3, bad.
- If c0==c1 and c2==c3 and c0!=c2, distinct is 2. Bad.
- If c0==c2 and c1==c3 and c0!=c1, distinct is 2. Bad.
- If c0==c3 and c1==c2 and c0!=c1, distinct is 2. Bad.

So, valid if:
NOT (
  (c0==c1 and c0==c2) or
  (c0==c1 and c0==c3) or
  (c0==c2 and c0==c3) or
  (c1==c2 and c1==c3) or
  (c0==c1 and c2==c3 and c0!=c2) or
  (c0==c2 and c1==c3 and c0!=c1) or
  (c0==c3 and c1==c2 and c0!=c1)
)

This is still a bit slow in Python.
However, `len({c0, c1, c2, c3})` is implemented in C and might be faster than Python logic for small sets. I will stick with `len` for simplicity unless profiling shows otherwise. Given N=128 and M~1000, and local search steps ~10k, total checks ~10M. Python `len(set)` is ~100ns? 10M * 100ns = 1s. Should be fine.

I will structure the code to try 3 colors first. If it fails (violations > 0 after search), try 4 colors.

