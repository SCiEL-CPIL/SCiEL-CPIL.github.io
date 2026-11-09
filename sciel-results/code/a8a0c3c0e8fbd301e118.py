"""
Strategy: Simulated Annealing (SA) with bitset representation.
We represent sets A and B as bitmasks (256 bits). 
Moves: 
1. Add a random element to A or B (if valid).
2. Remove a random element from A or B.
3. Swap an element in A with one in B (or vice versa) or within the same set.
4. Random restart if stuck for too long.

The energy function is simply -(|A| + |B|), so we want to minimize energy (maximize size).
Since we can only add valid elements, the search space is the set of valid configurations.
We use a "repair" approach: if a move makes the state invalid, we reject it (or revert).
Actually, for SA, it's better to define moves that always keep the state valid, 
or allow invalid states with high penalty. 
Given the strict constraints, keeping states valid is safer and faster to evaluate 
if we maintain incremental sums/diffs. However, maintaining incremental structures 
through swaps is complex. 
Alternative: Since N=256 is small, we can afford O(N^2) validation per move 
if we limit the number of moves. 
But SA needs many moves. 
Let's use a "greedy + SA" hybrid: 
Start with a good greedy solution. 
Then, try to improve by:
- Removing one element and trying to add two others (1-for-2 swap).
- Swapping one element in A with one in B to free up difference constraints.
- Randomly perturbing the set and re-greedy-filling.

Actually, a pure SA on the bitset with a fast validity check is risky if validity is slow.
Let's use a "Large Neighborhood Search" (LNS) approach:
1. Start with a valid solution from greedy.
2. Repeat:
   a. Remove k random elements (k=1,2,3).
   b. Re-run a fast greedy fill on the remaining structure, allowing the removed elements to be re-added or replaced by others.
   c. If the new solution is better, accept. Else, with probability exp(-delta/T), accept (for diversification, though usually we only accept improvements in maximization unless we have a penalty). 
   Since we maximize size, and size is integer, we can just accept improvements. To escape local optima, we can accept neutral moves (same size) with some probability, or use the 1-for-2 logic.
   
   A better structural change: **Iterated Local Search (ILS)**.
   1. Generate initial solution via randomized greedy.
   2. Local search: Try all possible single-element additions (should be none if greedy was exhaustive). 
      Try all possible "swaps": remove x from A, add y to A (if valid). 
      Try "cross-swaps": remove x from A, add y to B; remove z from B, add w to A.
   3. If no improvement, perturb: remove 2-3 random elements.
   4. Re-optimize with greedy.
   
   This is different from the current program which just does greedy + a weak "local search" that was essentially empty.
"""

import time
import random
import sys

N = 256
MAX_EACH = 64
DEADLINE_SECONDS = 85

def _check_sidon(s):
    """Return True if s is a Sidon set."""
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            val = s[i] + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def _diffs(s):
    """Return set of nonzero absolute differences."""
    d = set()
    n = len(s)
    for i in range(n):
        for j in range(i):
            d.add(s[i] - s[j])
    return d

def _valid_pair(A, B):
    if not _check_sidon(A) or not _check_sidon(B):
        return False
    dA = _diffs(A)
    dB = _diffs(B)
    return not (dA & dB)

def _greedy_fill(A, B, sums_A, sums_B, diffs_A, diffs_B, candidates=None):
    """
    Given current states, try to add more elements from candidates.
    Returns updated A, B, sums, diffs.
    """
    if candidates is None:
        candidates = set(range(1, N + 1))
    
    # Remove elements already in A or B from candidates? 
    # Actually, we want to add NEW elements. 
    # But we might want to re-add removed ones. 
    # Let's assume candidates is the pool of all numbers 1..N.
    # We check if x is already in A or B. If so, skip (unless we allow duplicates? No, sets).
    
    A = list(A)
    B = list(B)
    
    # Rebuild sums and diffs to be safe if they were passed partially
    # But for efficiency, we assume they are consistent.
    
    active = set(candidates)
    
    # Sort candidates randomly to introduce diversity
    order = list(active)
    random.shuffle(order)
    
    for x in order:
        if x in A or x in B:
            continue
            
        # Try to add to A
        ok_A = True
        if len(A) < MAX_EACH:
            if x + x in sums_A:
                ok_A = False
            else:
                for a in A:
                    if x + a in sums_A:
                        ok_A = False
                        break
            if ok_A:
                for a in A:
                    d = abs(x - a)
                    if d in diffs_A or d in diffs_B:
                        ok_A = False
                        break
        else:
            ok_A = False
            
        # Try to add to B
        ok_B = True
        if len(B) < MAX_EACH:
            if x + x in sums_B:
                ok_B = False
            else:
                for b in B:
                    if x + b in sums_B:
                        ok_B = False
                        break
            if ok_B:
                for b in B:
                    d = abs(x - b)
                    if d in diffs_B or d in diffs_A:
                        ok_B = False
                        break
        else:
            ok_B = False
            
        if ok_A:
            A.append(x)
            sums_A.add(x + x)
            for a in A[:-1]:
                sums_A.add(x + a)
                diffs_A.add(abs(x - a))
        elif ok_B:
            B.append(x)
            sums_B.add(x + x)
            for b in B[:-1]:
                sums_B.add(x + b)
                diffs_B.add(abs(x - b))
                
    return A, B, sums_A, sums_B, diffs_A, diffs_B

def _initial_greedy():
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    # Randomized greedy with multiple restarts to find a good starting point
    best_score = 0
    best_state = ([], [], set(), set(), set(), set())
    
    for _ in range(20):
        A, B, sums_A, sums_B, diffs_A, diffs_B = _greedy_fill([], [], set(), set(), set(), set())
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_state = (A, B, sums_A, sums_B, diffs_A, diffs_B)
            
    return best_state

def _local_search(A, B, sums_A, sums_B, diffs_A, diffs_B, deadline):
    """
    Try to improve the solution by single-element swaps or 1-for-2 additions.
    """
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try all possible single additions (should be none if greedy was exhaustive, but maybe after a swap?)
        # Actually, greedy in _greedy_fill is order-dependent. 
        # Let's try to add any x in 1..N not in A or B.
        
        # This is O(N * (|A|+|B|)) which is fine.
        for x in range(1, N + 1):
            if x in A or x in B:
                continue
                
            # Check if x can be added to A
            ok_A = True
            if len(A) < MAX_EACH:
                if x + x in sums_A:
                    ok_A = False
                else:
                    for a in A:
                        if x + a in sums_A:
                            ok_A = False
                            break
                if ok_A:
                    for a in A:
                        d = abs(x - a)
                        if d in diffs_A or d in diffs_B:
                            ok_A = False
                            break
            else:
                ok_A = False
                
            if ok_A:
                A.append(x)
                sums_A.add(x + x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                    diffs_A.add(abs(x - a))
                improved = True
                break # Restart loop to re-evaluate
                
            # Check if x can be added to B
            ok_B = True
            if len(B) < MAX_EACH:
                if x + x in sums_B:
                    ok_B = False
                else:
                    for b in B:
                        if x + b in sums_B:
                            ok_B = False
                            break
                if ok_B:
                    for b in B:
                        d = abs(x - b)
                        if d in diffs_B or d in diffs_A:
                            ok_B = False
                            break
            else:
                ok_B = False
                
            if ok_B:
                B.append(x)
                sums_B.add(x + x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                    diffs_B.add(abs(x - b))
                improved = True
                break # Restart loop
        
        if improved:
            continue
            
        # If no single addition possible, try swaps:
        # Remove x from A, add y to A (if valid)
        # Or remove x from A, add y to B
        # This is expensive. Let's try a limited number of random swaps.
        
        # Random perturbation: remove 1-2 elements and re-greedy
        if random.random() < 0.5:
            # Remove 1 random element from A or B
            if A:
                idx = random.randrange(len(A))
                rem = A[idx]
                # Rebuild sets after removal is complex. 
                # Easier: just mark it as "removed" and run a partial re-greedy?
                # No, let's just do a full re-greedy on the remaining elements + new candidates.
                
                # This is effectively a "kick" in ILS.
                A_new = [a for i, a in enumerate(A) if i != idx]
                # We don't easily remove from sums/diffs sets without rebuilding.
                # Rebuild is O(N^2). 256^2 = 65k. Acceptable for a few kicks.
                
                sums_A_new = set()
                diffs_A_new = set()
                for i in range(len(A_new)):
                    for j in range(i, len(A_new)):
                        sums_A_new.add(A_new[i] + A_new[j])
                    for j in range(i):
                        diffs_A_new.add(abs(A_new[i] - A_new[j]))
                        
                # Keep B, sums_B, diffs_B as is.
                # Run greedy fill on A_new, B with all candidates.
                A_res, B_res, sums_A_res, sums_B_res, diffs_A_res, diffs_B_res = _greedy_fill(
                    A_new, B, sums_A_new, sums_B, diffs_A_new, diffs_B
                )
                
                if len(A_res) + len(B_res) > len(A) + len(B):
                    A, B, sums_A, sums_B, diffs_A, diffs_B = A_res, B_res, sums_A_res, sums_B_res, diffs_A_res, diffs_B_res
                    improved = True
                    break
            elif B:
                idx = random.randrange(len(B))
                rem = B[idx]
                B_new = [b for i, b in enumerate(B) if i != idx]
                
                sums_B_new = set()
                diffs_B_new = set()
                for i in range(len(B_new)):
                    for j in range(i, len(B_new)):
                        sums_B_new.add(B_new[i] + B_new[j])
                    for j in range(i):
                        diffs_B_new.add(abs(B_new[i] - B_new[j]))
                        
                A_res, B_res, sums_A_res, sums_B_res, diffs_A_res, diffs_B_res = _greedy_fill(
                    A, B_new, sums_A, sums_B_new, diffs_A, diffs_B_new
                )
                
                if len(A_res) + len(B_res) > len(A) + len(B):
                    A, B, sums_A, sums_B, diffs_A, diffs_B = A_res, B_res, sums_A_res, sums_B_res, diffs_A_res, diffs_B_res
                    improved = True
                    break

    return A, B, sums_A, sums_B, diffs_A, diffs_B

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_A, best_B = [], []
    best_score = 0
    
    # 1. Get initial solution
    A, B, sums_A, sums_B, diffs_A, diffs_B = _initial_greedy()
    best_score = len(A) + len(B)
    best_A, best_B = list(A), list(B)
    
    # 2. Iterated Local Search
    while time.time() < deadline - 10:
        # Local search
        A, B, sums_A, sums_B, diffs_A, diffs_B = _local_search(
            A, B, sums_A, sums_B, diffs_A, diffs_B, deadline
        )
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A, best_B = list(A), list(B)
            
        # Perturbation: Randomly remove 2-3 elements and re-greedy
        if time.time() > deadline - 10:
            break
            
        k = random.choice([2, 3])
        
        # Randomly select elements to remove from A or B
        all_elements_A = list(range(len(A)))
        all_elements_B = list(range(len(B)))
        
        A_new = list(A)
        B_new = list(B)
        
        # Remove k elements randomly from the union, maintaining side integrity
        # Simpler: remove from A and B independently
        if len(A) >= k // 2:
            remove_from_A = random.sample(all_elements_A, min(k // 2, len(A)))
            for idx in sorted(remove_from_A, reverse=True):
                A_new.pop(idx)
        if len(B) >= k - len(remove_from_A):
            remove_from_B = random.sample(all_elements_B, min(k - len(remove_from_A), len(B)))
            for idx in sorted(remove_from_B, reverse=True):
                B_new.pop(idx)
                
        # Rebuild sums/diffs for the new sets
        sums_A_new = set()
        diffs_A_new = set()
        for i in range(len(A_new)):
            for j in range(i, len(A_new)):
                sums_A_new.add(A_new[i] + A_new[j])
            for j in range(i):
                diffs_A_new.add(abs(A_new[i] - A_new[j]))
                
        sums_B_new = set()
        diffs_B_new = set()
        for i in range(len(B_new)):
            for j in range(i, len(B_new)):
                sums_B_new.add(B_new[i] + B_new[j])
            for j in range(i):
                diffs_B_new.add(abs(B_new[i] - B_new[j]))
                
        # Re-greedy fill
        A_res, B_res, sums_A_res, sums_B_res, diffs_A_res, diffs_B_res = _greedy_fill(
            A_new, B_new, sums_A_new, sums_B_new, diffs_A_new, diffs_B_new
        )
        
        # Update current state (even if worse, to diversify? 
        # In ILS, we usually accept worse solutions to escape basins. 
        # But we must keep the BEST solution separately.)
        A, B, sums_A, sums_B, diffs_A, diffs_B = A_res, B_res, sums_A_res, sums_B_res, diffs_A_res, diffs_B_res
        
    # Final verification
    if not _valid_pair(best_A, best_B):
        # Fallback
        best_A = [1]
        best_B = [3]
        
    return [best_A, best_B]

if __name__ == "__main__":
    A, B = construct()
    print(f"A: {len(A)}, B: {len(B)}, Total: {len(A)+len(B)}")
    print(_valid_pair(A, B))

