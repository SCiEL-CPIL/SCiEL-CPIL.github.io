"""Structural change: Efficient Local Search with Incremental Score Updates and Random Perturbation."""
import time
import random

N = 256
TIME_LIMIT = 85.0

def construct():
    start_time = time.time()
    rng = random.Random(42)

    # Precompute 3-APs and 4-APs
    three_aps = []
    for d in range(1, (N - 2) // 2 + 1):
        for a in range(0, N - 2 * d):
            three_aps.append((a, a + d, a + 2 * d))

    four_aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(0, N - 3 * d):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    # Map element to 3-APs it participates in (store indices of other elements)
    elem_to_3aps = [[] for _ in range(N)]
    for a, b, c in three_aps:
        elem_to_3aps[a].append((b, c))
        elem_to_3aps[b].append((a, c))
        elem_to_3aps[c].append((a, b))

    # Map element to 4-APs it participates in (store indices of other elements)
    elem_to_4aps = [[] for _ in range(N)]
    for a, b, c, d in four_aps:
        elem_to_4aps[a].append((b, c, d))
        elem_to_4aps[b].append((a, c, d))
        elem_to_4aps[c].append((a, b, d))
        elem_to_4aps[d].append((a, b, c))

    # State
    in_set = bytearray(N)  # 0 or 1
    current_3ap_count = 0
    current_4ap_count = 0
    
    # For faster violation tracking, we can just recompute 4-AP count if needed, 
    # but since N is small, checking safety is the bottleneck.
    
    def is_safe_add(x):
        """Check if adding x creates any 4-AP."""
        for others in elem_to_4aps[x]:
            if in_set[others[0]] and in_set[others[1]] and in_set[others[2]]:
                return False
        return True

    def gain_3aps(x):
        """Calculate gain in 3-AP count if x is added."""
        gain = 0
        for o1, o2 in elem_to_3aps[x]:
            if in_set[o1] and in_set[o2]:
                gain += 1
        return gain

    def loss_3aps(x):
        """Calculate loss in 3-AP count if x is removed."""
        loss = 0
        for o1, o2 in elem_to_3aps[x]:
            if in_set[o1] and in_set[o2]:
                loss += 1
        return loss

    def count_3aps_total():
        """Recompute total 3-AP count (used for verification or init)."""
        cnt = 0
        for a, b, c in three_aps:
            if in_set[a] and in_set[b] and in_set[c]:
                cnt += 1
        return cnt

    def count_4aps_total():
        """Recompute total 4-AP count."""
        cnt = 0
        for a, b, c, d in four_aps:
            if in_set[a] and in_set[b] and in_set[c] and in_set[d]:
                cnt += 1
        return cnt

    def greedy_init():
        """Initialize with a greedy strategy: add elements with high 3-AP potential, skip if unsafe."""
        nonlocal in_set, current_3ap_count
        in_set = bytearray(N)
        current_3ap_count = 0
        
        # Sort elements by potential 3-AP count (degree in 3-AP graph)
        weights = [len(elem_to_3aps[x]) for x in range(N)]
        order = sorted(range(N), key=lambda x: (-weights[x], x))
        
        for x in order:
            if is_safe_add(x):
                in_set[x] = 1
                current_3ap_count += gain_3aps(x)
        
        # Ensure no 4-APs (should be none by construction, but double check)
        # If any exist, remove elements to break them
        while count_4aps_total() > 0:
            # Find a 4-AP and remove one element
            removed = False
            for a, b, c, d in four_aps:
                if in_set[a] and in_set[b] and in_set[c] and in_set[d]:
                    # Remove the one with highest loss? Or just first?
                    # Let's remove the one with lowest 3-AP contribution to minimize damage
                    losses = [loss_3aps(a), loss_3aps(b), loss_3aps(c), loss_3aps(d)]
                    min_idx = min(range(4), key=lambda i: losses[i])
                    elems = [a, b, c, d]
                    x_to_remove = elems[min_idx]
                    in_set[x_to_remove] = 0
                    current_3ap_count -= losses[min_idx]
                    removed = True
                    break
            if not removed:
                break

    def local_search(max_iters=500):
        """Local search: try adds and swaps to improve 3-AP count while maintaining 4-AP-free."""
        nonlocal in_set, current_3ap_count
        
        members = [x for x in range(N) if in_set[x]]
        non_members = [x for x in range(N) if not in_set[x]]
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # 1. Try single adds
            # Sort non-members by potential gain to find best add quickly?
            # For now, iterate.
            best_add = -1
            best_gain = 0
            for x in non_members:
                if not is_safe_add(x):
                    continue
                g = gain_3aps(x)
                if g > best_gain:
                    best_gain = g
                    best_add = x
            
            if best_add != -1 and best_gain > 0:
                in_set[best_add] = 1
                current_3ap_count += best_gain
                members.append(best_add)
                non_members.remove(best_add)
                improved = True
                continue

            # 2. Try swaps (remove one, add one)
            # This is O(M * NM) which might be slow. 
            # Optimization: Only try swaps if we are stuck.
            best_swap = None
            best_net = 0
            # Limit swap search to a subset if too slow? 
            # N=256, M~100, NM~150. 100*150 = 15000 checks. Each check is O(1) roughly.
            # 15000 * 500 iters = 7.5M ops. Should be fast enough in Python if optimized.
            
            for x_rm in members:
                # Temporarily remove x_rm
                in_set[x_rm] = 0
                loss = loss_3aps(x_rm)
                
                for x_add in non_members:
                    if not is_safe_add(x_add):
                        continue
                    gain = gain_3aps(x_add)
                    net = gain - loss
                    if net > best_net:
                        best_net = net
                        best_swap = (x_rm, x_add)
                
                # Restore x_rm for next iteration of outer loop
                in_set[x_rm] = 1

            if best_swap is not None and best_net > 0:
                x_rm, x_add = best_swap
                in_set[x_rm] = 0
                in_set[x_add] = 1
                current_3ap_count += best_net
                members.remove(x_rm)
                non_members.remove(x_add)
                members.append(x_add)
                improved = True
                continue

            # 3. If stuck, do a random perturbation
            if iters % 20 == 0:
                # Remove 2-3 random members, add 2-3 random non-members (if safe)
                for _ in range(rng.randint(2, 4)):
                    if members:
                        x_rm = rng.choice(members)
                        in_set[x_rm] = 0
                        current_3ap_count -= loss_3aps(x_rm)
                        members.remove(x_rm)
                        non_members.append(x_rm)
                
                for _ in range(rng.randint(2, 4)):
                    if non_members:
                        x_add = rng.choice(non_members)
                        if is_safe_add(x_add):
                            in_set[x_add] = 1
                            current_3ap_count += gain_3aps(x_add)
                            non_members.remove(x_add)
                            members.append(x_add)
                improved = True # Force another loop to re-evaluate

        return current_3ap_count

    # Main Loop
    greedy_init()
    best_score = current_3ap_count
    best_set = in_set.copy()

    deadline = start_time + TIME_LIMIT - 2.0
    
    # Phase 1: Local Search on Greedy
    local_search(max_iters=200)
    if current_3ap_count > best_score:
        best_score = current_3ap_count
        best_set = in_set.copy()

    # Phase 2: Multi-start with Perturbation
    while time.time() < deadline:
        # Perturb the current best
        in_set = best_set.copy()
        current_3ap_count = best_score
        
        # Randomly flip some bits
        num_flips = rng.randint(5, 15)
        for _ in range(num_flips):
            x = rng.randint(0, N - 1)
            if in_set[x]:
                in_set[x] = 0
                current_3ap_count -= loss_3aps(x)
            else:
                # Check safety before adding
                if is_safe_add(x):
                    in_set[x] = 1
                    current_3ap_count += gain_3aps(x)
        
        # Repair: remove elements to break any 4-APs created by perturbation
        # Note: Perturbation only adds if safe, so no new 4-APs should be created by adds.
        # But removals might break 3-APs. 
        # Wait, if we remove an element, we don't create 4-APs.
        # If we add an element, we checked safety. So the set should still be 4-AP free.
        # However, the "safety" check depends on the current state.
        # If we remove x, then add y, y's safety was checked against state WITH x?
        # No, we check is_safe_add(y) against current in_set.
        # So the set remains 4-AP free.
        
        # Run local search
        local_search(max_iters=100)
        
        if current_3ap_count > best_score:
            best_score = current_3ap_count
            best_set = in_set.copy()

    # Final Verification
    final_set = best_set
    # Ensure no 4-APs
    while count_4aps_total() > 0:
        for a, b, c, d in four_aps:
            if final_set[a] and final_set[b] and final_set[c] and final_set[d]:
                # Remove one
                final_set[a] = 0
                break

    # Convert to list
    result = [x + 1 for x in range(N) if final_set[x]]
    
    if not result:
        # Fallback
        result = [1]

    return result
