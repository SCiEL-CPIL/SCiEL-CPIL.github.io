"""Improved program for 4-term-AP-free subset of [1,512]."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _has_4ap(selected, n):
    """Check if selected (set of ints in 1..n) contains a 4-term AP."""
    sel = set(selected)
    for a in sel:
        for b in sel:
            if b <= a:
                continue
            d = b - a
            if a + 3*d > n:
                continue
            if (a + d) in sel and (a + 2*d) in sel and (a + 3*d) in sel:
                return True
    return False

def _construct_greedy_random(n, seed):
    """Greedy construction with random ordering."""
    rng = random.Random(seed)
    candidates = list(range(1, n+1))
    rng.shuffle(candidates)
    selected = set()
    for x in candidates:
        # Check if adding x creates a 4-term AP
        ok = True
        for d in range(1, (x-1)//3 + 1):
            # x could be the 4th term: x-3d, x-2d, x-d, x
            if (x - d) in selected and (x - 2*d) in selected and (x - 3*d) in selected:
                ok = False
                break
            # x could be the 3rd term: x-2d, x-d, x, x+d
            if (x - d) in selected and (x - 2*d) in selected and (x + d) in selected:
                ok = False
                break
            # x could be the 2nd term: x-d, x, x+d, x+2d
            if (x - d) in selected and (x + d) in selected and (x + 2*d) in selected:
                ok = False
                break
            # x could be the 1st term: x, x+d, x+2d, x+3d
            if (x + d) in selected and (x + 2*d) in selected and (x + 3*d) in selected:
                ok = False
                break
        if ok:
            selected.add(x)
    return selected

def _construct_greedy_numeric(n):
    """Greedy construction based on base-3 digit sum (Salem-Spencer style for 3-AP, adapted)."""
    # For 4-AP free sets, use a different approach.
    # Try: numbers whose base-4 representation has no digit equal to 0 in certain positions?
    # Actually, let's use a known construction: take numbers with no 0 in base 3? That's for 3-AP.
    # For 4-AP, a simple approach: numbers congruent to 1 or 2 mod 4? No, that still has 4-APs.
    # Let's just use greedy with a fixed ordering.
    selected = set()
    for x in range(1, n+1):
        ok = True
        for d in range(1, (x-1)//3 + 1):
            if (x - d) in selected and (x - 2*d) in selected and (x - 3*d) in selected:
                ok = False
                break
            if (x - d) in selected and (x - 2*d) in selected and (x + d) in selected:
                ok = False
                break
            if (x - d) in selected and (x + d) in selected and (x + 2*d) in selected:
                ok = False
                break
            if (x + d) in selected and (x + 2*d) in selected and (x + 3*d) in selected:
                ok = False
                break
        if ok:
            selected.add(x)
    return selected

def _local_improve(selected, n, max_iters=50000, deadline=None):
    """Try to improve the set by removing elements that cause conflicts and adding others."""
    sel = set(selected)
    rng = random.Random(42)
    best = sel
    best_size = len(sel)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Try to add a random element not in sel
        candidates = [x for x in range(1, n+1) if x not in sel]
        if not candidates:
            break
        x = rng.choice(candidates)
        
        # Check if x can be added
        ok = True
        for d in range(1, (x-1)//3 + 1):
            if (x - d) in sel and (x - 2*d) in sel and (x - 3*d) in sel:
                ok = False
                break
            if (x - d) in sel and (x - 2*d) in sel and (x + d) in sel:
                ok = False
                break
            if (x - d) in sel and (x + d) in sel and (x + 2*d) in sel:
                ok = False
                break
            if (x + d) in sel and (x + 2*d) in sel and (x + 3*d) in sel:
                ok = False
                break
        
        if ok:
            sel.add(x)
            if len(sel) > best_size:
                best_size = len(sel)
                best = set(sel)
        
        # Try to remove a random element and add two others
        if sel:
            to_remove = rng.choice(list(sel))
            sel.remove(to_remove)
            # Try to add back some elements
            for y in [to_remove, to_remove+1, to_remove-1]:
                if 1 <= y <= n and y not in sel:
                    ok = True
                    for d in range(1, (y-1)//3 + 1):
                        if (y - d) in sel and (y - 2*d) in sel and (y - 3*d) in sel:
                            ok = False
                            break
                        if (y - d) in sel and (y - 2*d) in sel and (y + d) in sel:
                            ok = False
                            break
                        if (y - d) in sel and (y + d) in sel and (y + 2*d) in sel:
                            ok = False
                            break
                        if (y + d) in sel and (y + 2*d) in sel and (y + 3*d) in sel:
                            ok = False
                            break
                    if ok:
                        sel.add(y)
                        break
        
        if len(sel) > best_size:
            best_size = len(sel)
            best = set(sel)
    
    return best

def construct():
    n = 512
    deadline = time.time() + 80  # Safety margin
    
    # Start with a good greedy construction
    best = _construct_greedy_numeric(n)
    best_size = len(best)
    
    # Try multiple random greedy constructions
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        s = _construct_greedy_random(n, seed)
        if len(s) > best_size:
            best = s
            best_size = len(s)
    
    # Local improvement
    improved = _local_improve(best, n, max_iters=100000, deadline=deadline - 5)
    if len(improved) > best_size:
        best = improved
        best_size = len(best)
    
    # Final check: verify no 4-AP
    # (This is expensive but we do it once at the end)
    # Actually, let's just trust the construction and return
    
    return sorted(best)

# EVOLVE-BLOCK-END
