"""Combined: precomputed comparability matrix + exhaustive hill climbing + multi-restart SA."""
import time
import random

_start_time = time.time()

def construct():
    deadline = _start_time + 85
    n = 12
    m = 40
    total_masks = 1 << n  # 4096

    # Precompute comparability matrix using bytearray for memory efficiency
    comp = [bytearray(total_masks) for _ in range(total_masks)]
    for a in range(total_masks):
        for b in range(a + 1, total_masks):
            if (a | b) == a or (a | b) == b:
                comp[a][b] = 1
                comp[b][a] = 1

    def count_comparables(family):
        cnt = 0
        for i in range(len(family)):
            fi = family[i]
            for j in range(i + 1, len(family)):
                if comp[fi][family[j]]:
                    cnt += 1
        return cnt

    def get_degrees(family):
        degrees = [0] * len(family)
        for i in range(len(family)):
            fi = family[i]
            for j in range(i + 1, len(family)):
                if comp[fi][family[j]]:
                    degrees[i] += 1
                    degrees[j] += 1
        return degrees

    def make_initial_family():
        """Create initial family using chains + greedy completion."""
        family = []
        seen = set()

        # Chain 1: {0}, {0,1}, {0,1,2}, ..., {0..11} (13 sets including empty)
        for sz in range(n + 1):
            mask = (1 << sz) - 1
            family.append(mask)
            seen.add(mask)

        # Chain 2: {1}, {1,2}, ..., {1..11,0} - add only new sets
        s = 0
        for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]:
            s |= (1 << v)
            if s not in seen:
                family.append(s)
                seen.add(s)

        # Chain 3: {2}, {2,3}, ..., {2..11,0,1}
        s = 0
        for v in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1]:
            s |= (1 << v)
            if s not in seen:
                family.append(s)
                seen.add(s)

        # Greedily add remaining sets to maximize comparability
        all_masks = list(range(total_masks))
        random.shuffle(all_masks)

        while len(family) < m:
            best_add = None
            best_deg = -1
            for cand in all_masks:
                if cand in seen:
                    continue
                deg = 0
                for s in family:
                    if comp[cand][s]:
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_add = cand
            if best_add is not None:
                family.append(best_add)
                seen.add(best_add)
            else:
                for cand in all_masks:
                    if cand not in seen:
                        family.append(cand)
                        seen.add(cand)
                        break
                if len(family) >= m:
                    break

        return family[:m]

    def exhaustive_hill_climb(family, time_limit):
        """Exhaustive hill climbing: for each slot, try all 4096 masks."""
        seen = set(family)
        current_score = count_comparables(family)
        best_family = list(family)
        best_score = current_score

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            degrees = get_degrees(family)

            for i in range(len(family)):
                if time.time() > time_limit:
                    break
                old_set = family[i]
                old_deg = degrees[i]

                best_new = None
                best_gain = 0

                for new_set in range(total_masks):
                    if new_set == old_set or new_set in seen:
                        continue

                    new_deg = 0
                    for j in range(len(family)):
                        if j == i:
                            continue
                        if comp[new_set][family[j]]:
                            new_deg += 1

                    gain = new_deg - old_deg
                    if gain > best_gain:
                        best_gain = gain
                        best_new = new_set

                if best_new is not None and best_gain > 0:
                    family[i] = best_new
                    seen.remove(old_set)
                    seen.add(best_new)
                    current_score += best_gain
                    improved = True

                    if current_score > best_score:
                        best_score = current_score
                        best_family = list(family)

                    break  # Restart scan from beginning after a change

        return best_family, best_score

    def simulated_annealing(family, start_temp, end_temp, cooling, time_limit):
        """Simulated annealing to escape local optima."""
        seen = set(family)
        current_score = count_comparables(family)
        best_family = list(family)
        best_score = current_score
        temp = start_temp

        while temp > end_temp and time.time() < time_limit:
            i = random.randint(0, len(family) - 1)
            old = family[i]
            
            new = random.randint(0, total_masks - 1)
            if new == old or new in seen:
                temp *= cooling
                continue

            # Compute degrees
            old_deg = 0
            new_deg = 0
            for j in range(len(family)):
                if j == i:
                    continue
                if comp[old][family[j]]:
                    old_deg += 1
                if comp[new][family[j]]:
                    new_deg += 1

            delta = new_deg - old_deg
            if delta > 0 or random.random() < pow(2.718281828, delta / temp):
                family[i] = new
                seen.remove(old)
                seen.add(new)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_family = list(family)

            temp *= cooling

        return best_family, best_score

    random.seed(42)

    # Multi-restart: try different initializations and search strategies
    best_family_overall = None
    best_score_overall = 0

    # Restart 1: Chain-based init + exhaustive hill climb
    family1 = make_initial_family()
    time_limit1 = deadline - 40
    best_f1, best_s1 = exhaustive_hill_climb(family1, time_limit1)
    best_family_overall = best_f1
    best_score_overall = best_s1

    # Restart 2: SA from best of restart 1, then hill climb again
    if time.time() < deadline - 25:
        family2 = list(best_f1)
        time_limit2 = deadline - 20
        best_f2, best_s2 = simulated_annealing(family2, 5.0, 0.1, 0.95, time_limit2)
        if best_s2 > best_score_overall:
            best_family_overall = best_f2
            best_score_overall = best_s2

        # Hill climb from SA result
        if time.time() < deadline - 5:
            family3 = list(best_f2)
            time_limit3 = deadline - 3
            best_f3, best_s3 = exhaustive_hill_climb(family3, time_limit3)
            if best_s3 > best_score_overall:
                best_family_overall = best_f3
                best_score_overall = best_s3

    # Restart 3: Different chain ordering + hill climb
    if time.time() < deadline - 15:
        # Try a different initial family
        family4 = []
        seen4 = set()
        # Reverse chain
        for sz in range(n, -1, -1):
            mask = (1 << sz) - 1
            if mask not in seen4:
                family4.append(mask)
                seen4.add(mask)
        # Another chain
        s = 0
        for v in [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]:
            s |= (1 << v)
            if s not in seen4:
                family4.append(s)
                seen4.add(s)
        # Greedy completion
        all_masks = list(range(total_masks))
        random.shuffle(all_masks)
        while len(family4) < m:
            best_add = None
            best_deg = -1
            for cand in all_masks:
                if cand in seen4:
                    continue
                deg = 0
                for s in family4:
                    if comp[cand][s]:
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_add = cand
            if best_add is not None:
                family4.append(best_add)
                seen4.add(best_add)
            else:
                for cand in all_masks:
                    if cand not in seen4:
                        family4.append(cand)
                        seen4.add(cand)
                        break
                if len(family4) >= m:
                    break
        family4 = family4[:m]

        time_limit4 = deadline - 5
        best_f4, best_s4 = exhaustive_hill_climb(family4, time_limit4)
        if best_s4 > best_score_overall:
            best_family_overall = best_f4
            best_score_overall = best_s4

    # Final polish: one more exhaustive pass on the best family
    if time.time() < deadline - 2 and best_family_overall is not None:
        family_final = list(best_family_overall)
        time_limit_final = deadline - 1
        best_f_final, best_s_final = exhaustive_hill_climb(family_final, time_limit_final)
        if best_s_final > best_score_overall:
            best_family_overall = best_f_final
            best_score_overall = best_s_final

    # Convert to lists of ints
    result = []
    for mask in best_family_overall:
        lst = [int(i) for i in range(n) if (mask >> i) & 1]
        result.append(lst)

    return result

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    assert len(res) == 40
    seen = set()
    for r in res:
        fr = frozenset(r)
        assert fr not in seen
        seen.add(fr)
        for v in r:
            assert isinstance(v, int)
            assert 0 <= v <= 11
    masks = []
    for r in res:
        m = 0
        for v in r:
            m |= (1 << v)
        masks.append(m)
    cnt = 0
    for i in range(40):
        for j in range(i+1, 40):
            a, b = masks[i], masks[j]
            if a | b == a or a | b == b:
                cnt += 1
    print(f"Time: {t1-t0:.2f}s, Comparables: {cnt}")
