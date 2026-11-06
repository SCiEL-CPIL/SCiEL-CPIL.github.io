"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
N = 256

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a small inclusion-maximal Sidon set in [1, N].
    Uses a GRASP (Greedy Randomized Adaptive Search Procedure) heuristic:
    iteratively select a vertex that blocks many other candidates,
    with randomization among top candidates to explore different solutions.
    """
    best_selected = None
    best_size = float('inf')
    
    # Run multiple iterations to find a smaller set
    for _ in range(50):
        selected = []
        sums = set()
        available = set(range(1, N + 1))
        
        while available:
            candidates = []
            
            # Evaluate each candidate in 'available'
            for v in available:
                # 1. Check if v is compatible with current 'selected'
                new_sums_v = {2 * v}
                for s in selected:
                    new_sums_v.add(v + s)
                
                # If any new sum collides with existing sums, v is invalid
                if sums.intersection(new_sums_v):
                    continue
                
                # v is a valid candidate to add.
                # Calculate 'degree': how many other available elements u
                # become incompatible if we add v?
                
                # Efficiently find blocked u's using the conflict conditions:
                # u is blocked if:
                # 1. 2u == 2v (u=v, not in available)
                # 2. 2u == v+s  => u = (v+s)/2
                # 3. u+s == 2v  => u = 2v-s
                # 4. u+s == v+s' => u = v+s'-s
                # 5. u+v == s'+s'' => u = s'+s''-v
                
                blocked = set()
                
                # Condition 2: u = (v+s)/2
                for s in selected:
                    val = v + s
                    if val % 2 == 0:
                        u = val // 2
                        if u in available:
                            blocked.add(u)
                
                # Condition 3: u = 2v - s
                for s in selected:
                    u = 2 * v - s
                    if u in available:
                        blocked.add(u)
                
                # Condition 4: u = v + s' - s
                for s in selected:
                    for sp in selected:
                        u = v + sp - s
                        if u in available:
                            blocked.add(u)
                
                # Condition 5: u = s' + s'' - v
                # Precompute pair sums of selected to speed up?
                # For now, nested loop is fine for small |selected|
                for s in selected:
                    for sp in selected:
                        u = s + sp - v
                        if u in available:
                            blocked.add(u)
                
                # Remove v itself from blocked if it was added (shouldn't be, but safe)
                blocked.discard(v)
                
                degree = len(blocked)
                candidates.append((degree, v))
            
            if not candidates:
                # No valid vertex, set is maximal
                break
            
            # Sort by degree descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # GRASP: Pick from top K candidates
            # K is dynamic, e.g., top 10% or at least 1
            k = max(1, len(candidates) // 10)
            # To ensure diversity, we can pick randomly from the top k
            # But if we always pick the best, it's deterministic.
            # Let's pick randomly from the top k.
            chosen = random.choice(candidates[:k])
            best_v = chosen[1]
            
            # Update state with best_v
            selected.append(best_v)
            
            # Update sums
            sums.add(2 * best_v)
            for s in selected[:-1]:
                sums.add(best_v + s)
            
            # Update available: remove best_v and any u incompatible with new selected
            new_available = set()
            for u in available:
                if u == best_v:
                    continue
                
                # Check if u is compatible with new selected
                # New sums for u: 2u, u+s for s in selected
                # Collision if any in sums
                
                if 2 * u in sums:
                    continue
                
                ok = True
                for s in selected:
                    if (u + s) in sums:
                        ok = False
                    break
                # Note: The loop above has a break inside if ok becomes False?
                # No, the original code had a break. Let's fix the logic.
                # Actually, the 'ok' logic in the original code was:
                # ok = True
                # for s in selected:
                #    if (u+s) in sums: ok=False; break
                # So I need to replicate that.
                
                # Re-doing the check correctly:
                ok = True
                for s in selected:
                    if (u + s) in sums:
                        ok = False
                        break
                
                if ok:
                    new_available.add(u)
            
            available = new_available
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = selected[:]
    
    return best_selected if best_selected else [1]
# EVOLVE-BLOCK-END
