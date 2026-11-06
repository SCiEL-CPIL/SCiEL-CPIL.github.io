import time
import random

def construct():
    n = 512
    deadline = time.time() + 50.0  # Safety margin
    
    # Precompute all 4-APs to speed up checks?
    # There are O(n^2) APs.
    # APs: (a, a+d, a+2d, a+3d)
    all_aps = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    # For fast lookup, we can't just use a set of tuples easily for "does x conflict"
    # but we can check conflicts by iterating APs containing x?
    # No, `try_add` iterating d is fine.

    def try_add(in_set, x):
        # Check if adding x creates a 4-AP
        # x is part of AP (u, u+d, u+2d, u+3d)
        # 1. x = u
        for d in range(1, (n - x) // 3 + 1):
            if in_set[x + d] and in_set[x + 2 * d] and in_set[x + 3 * d]:
                return False
        # 2. x = u+d  => u = x-d
        for d in range(1, x):
            if x - d >= 1:
                if in_set[x - d] and in_set[x + d] and in_set[x + 2 * d] and (x + 2 * d <= n):
                    return False
            else:
                break
        # 3. x = u+2d => u = x-2d
        for d in range(1, (x - 1) // 2 + 1):
            if x - 2 * d >= 1:
                if in_set[x - 2 * d] and in_set[x - d] and in_set[x + d] and (x + d <= n):
                    return False
            else:
                break
        # 4. x = u+3d => u = x-3d
        for d in range(1, (x - 1) // 3 + 1):
            if x - 3 * d >= 1:
                if in_set[x - 3 * d] and in_set[x - 2 * d] and in_set[x - d]:
                    return False
            else:
                break
        return True

    best_size = 0
    best_set = set()
    
    # Run multiple random restarts
    while time.time() < deadline:
        # Initialize with a random subset
        in_set = [False] * (n + 1)
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        # Start with a higher density, e.g., 60%
        start_count = int(n * 0.6)
        for x in candidates[:start_count]:
            in_set[x] = True
        
        # Repair phase
        # We need to remove elements until no 4-AP exists.
        # To speed up, we can check all APs and mark "bad" elements.
        # If an element is in any 4-AP, it's a candidate for removal.
        # Let's iterate:
        # 1. Find all APs that are fully in `in_set`.
        # 2. If none, break.
        # 3. Pick one AP, remove one element.
        
        repair_iter = 0
        while time.time() < deadline:
            # Find a 4-AP
            found_ap = None
            for a, b, c, d in all_aps:
                if in_set[a] and in_set[b] and in_set[c] and in_set[d]:
                    found_ap = (a, b, c, d)
                    break
            
            if found_ap is None:
                break
            
            # Remove a random element from this AP
            # Removing the middle one (c) is often effective
            # Let's pick random index
            idx = random.randint(0, 3)
            elem_to_remove = found_ap[idx]
            in_set[elem_to_remove] = False
            
            repair_iter += 1
            if repair_iter > 1000:
                # Safety break if stuck in a weird state (unlikely)
                break
        
        # Optimization phase: Add back elements
        remaining = [x for x in range(1, n + 1) if not in_set[x]]
        random.shuffle(remaining)
        
        # Greedy add
        while time.time() < deadline:
            added_any = False
            # Iterate through remaining, try to add
            # We need to be careful with removing from list while iterating
            # Let's use a pointer or create a new list
            i = 0
            while i < len(remaining):
                x = remaining[i]
                if in_set[x]:
                    i += 1
                    continue
                if try_add(in_set, x):
                    in_set[x] = True
                    remaining.pop(i)
                    added_any = True
                    # Restart scan to catch new opportunities created by this add?
                    # Actually, adding an element only creates constraints, it doesn't
                    # enable other elements to be added if they were blocked before.
                    # Wait, if x was blocked by y, and y is still there, x is still blocked.
                    # Adding x doesn't unblock z.
                    # So we don't need to restart, just continue.
                    # However, if we removed elements in repair, the "remaining" list
                    # might contain elements that are now addable.
                    # Since we only add here, and don't remove, we can just continue.
                else:
                    i += 1
            
            if not added_any:
                # Stuck. Perturb.
                current_elements = [x for x in range(1, n + 1) if in_set[x]]
                if not current_elements:
                    break
                # Remove 1-5 random elements
                num_remove = random.randint(1, 5)
                for _ in range(num_remove):
                    if current_elements:
                        rm = random.choice(current_elements)
                        in_set[rm] = False
                        if rm not in remaining:
                            remaining.append(rm)
                random.shuffle(remaining)
                
        current_size = sum(in_set)
        if current_size > best_size:
            best_size = current_size
            best_set = set(x for x in range(1, n + 1) if in_set[x])
            
            if best_size >= 180:
                break

    return sorted(best_set)
