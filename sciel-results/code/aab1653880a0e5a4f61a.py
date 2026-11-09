"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80.0
    TARGET = 200
    MASK = (1 << (TARGET + 1)) - 1  # bits 0..200

    def reachable_mask(s):
        r = 1  # bit 0 set
        for x in s:
            r |= (r << x)
            r &= MASK
        return r

    def can_add(reach, x):
        new = (reach | (reach << x)) & MASK
        return (new >> TARGET) & 1 == 0, new

    # Greedy: iterate from high to low
    selected = set()
    reach = 1
    for x in range(100, 0, -1):
        ok, new_reach = can_add(reach, x)
        if ok:
            selected.add(x)
            reach = new_reach

    # Try to add excluded elements (multiple passes)
    excluded = [x for x in range(1, 101) if x not in selected]
    for _ in range(5):
        if time.time() > deadline - 5.0:
            break
        changed = False
        for x in excluded:
            if time.time() > deadline - 5.0:
                break
            if x in selected:
                continue
            ok, new_reach = can_add(reach, x)
            if ok:
                selected.add(x)
                reach = new_reach
                changed = True
        if not changed:
            break

    # 1-for-1 swap: remove one, add one excluded
    if time.time() < deadline - 10.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            excluded_list = [x for x in range(1, 101) if x not in selected]
            for rem in list(selected):
                if time.time() > deadline - 5.0:
                    break
                temp_reach = reachable_mask(selected - {rem})
                for add in excluded_list:
                    if time.time() > deadline - 5.0:
                        break
                    if add in selected:
                        continue
                    ok, _ = can_add(temp_reach, add)
                    if ok:
                        selected.discard(rem)
                        selected.add(add)
                        reach = reachable_mask(selected)
                        improved = True
                        break
                if improved:
                    break

    # 1-for-2 swaps: remove one, add two excluded
    if time.time() < deadline - 15.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            excluded_list = [x for x in range(1, 101) if x not in selected]
            for rem in list(selected):
                if time.time() > deadline - 5.0:
                    break
                temp_reach = reachable_mask(selected - {rem})
                added = []
                for add in excluded_list:
                    if time.time() > deadline - 5.0:
                        break
                    if add in selected:
                        continue
                    ok, new_reach = can_add(temp_reach, add)
                    if ok:
                        temp_reach = new_reach
                        added.append(add)
                if len(added) >= 2:
                    selected.discard(rem)
                    for a in added:
                        selected.add(a)
                    reach = reachable_mask(selected)
                    improved = True
                    break

    # 2-for-1 swap: remove two, add one excluded (net -1, but might enable other additions)
    # Actually this reduces size, so skip unless it enables larger gains later
    # 2-for-3 swaps: remove two, add three excluded
    if time.time() < deadline - 10.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            excluded_list = [x for x in range(1, 101) if x not in selected]
            sel_list = list(selected)
            for i in range(len(sel_list)):
                if time.time() > deadline - 5.0:
                    break
                for j in range(i + 1, len(sel_list)):
                    if time.time() > deadline - 5.0:
                        break
                    rem1, rem2 = sel_list[i], sel_list[j]
                    if rem1 not in selected or rem2 not in selected:
                        continue
                    temp_reach = reachable_mask(selected - {rem1, rem2})
                    added = []
                    for add in excluded_list:
                        if time.time() > deadline - 5.0:
                            break
                        if add in selected:
                            continue
                        ok, new_reach = can_add(temp_reach, add)
                        if ok:
                            temp_reach = new_reach
                            added.append(add)
                    if len(added) >= 3:
                        selected.discard(rem1)
                        selected.discard(rem2)
                        for a in added:
                            selected.add(a)
                        reach = reachable_mask(selected)
                        improved = True
                        break
                if improved:
                    break

    # Final: try adding any remaining excluded elements
    for x in range(1, 101):
        if time.time() > deadline - 2.0:
            break
        if x in selected:
            continue
        ok, new_reach = can_add(reach, x)
        if ok:
            selected.add(x)
            reach = new_reach

    return list(selected)
# EVOLVE-BLOCK-END
