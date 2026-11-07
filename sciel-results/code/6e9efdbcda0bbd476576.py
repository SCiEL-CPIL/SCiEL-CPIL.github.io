import time
import random

P = {'n': 12, 'm': 40}

def construct():
    deadline = time.time() + 80
    n = P['n']
    m = P['m']
    N = 1 << n

    # Precompute comparability table
    comp = [[False] * N for _ in range(N)]
    for a in range(N):
        for b in range(N):
            if a != b and (a & b) == a:
                comp[a][b] = True

    def count_comparable(family):
        count = 0
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                if comp[family[i]][family[j]] or comp[family[j]][family[i]]:
                    count += 1
        return count

    def local_search(family, max_iters, deadline):
        current_score = count_comparable(family)
        best_score = current_score
        best_family = family[:]
        no_improve = 0
        threshold = 3000

        for iteration in range(max_iters):
            if time.time() > deadline - 2:
                break

            move_type = random.random()

            if move_type < 0.7:
                # Single replacement
                i = random.randint(0, m-1)
                if random.random() < 0.5:
                    base = family[random.randint(0, m-1)]
                    bits = [b for b in range(n) if not (base >> b) & 1]
                    if bits and random.random() < 0.5:
                        new_mask = base | (1 << random.choice(bits))
                    else:
                        set_bits = [b for b in range(n) if (base >> b) & 1]
                        if not set_bits:
                            new_mask = random.randint(0, N-1)
                        else:
                            new_mask = base & ~(1 << random.choice(set_bits))
                else:
                    new_mask = random.randint(0, N-1)

                if new_mask == family[i]:
                    continue
                if any(new_mask == family[j] for j in range(m) if j != i):
                    continue

                old_mask = family[i]

                lost = 0
                gained = 0
                for j in range(m):
                    if j == i:
                        continue
                    other = family[j]
                    if comp[old_mask][other] or comp[other][old_mask]:
                        lost += 1
                    if comp[new_mask][other] or comp[other][new_mask]:
                        gained += 1

                delta = gained - lost

                if delta > 0:
                    family[i] = new_mask
                    current_score += delta
                    no_improve = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_family = family[:]
                else:
                    no_improve += 1

            else:
                # Two-swap
                idxs = random.sample(range(m), 2)
                i, j = idxs[0], idxs[1]
                old_i, old_j = family[i], family[j]

                if random.random() < 0.5:
                    new_i = random.randint(0, N-1)
                    new_j = random.randint(0, N-1)
                else:
                    base1 = family[random.randint(0, m-1)]
                    base2 = family[random.randint(0, m-1)]
                    bits1 = [b for b in range(n) if not (base1 >> b) & 1]
                    bits2 = [b for b in range(n) if not (base2 >> b) & 1]
                    if bits1:
                        new_i = base1 | (1 << random.choice(bits1))
                    else:
                        new_i = random.randint(0, N-1)
                    if bits2:
                        new_j = base2 | (1 << random.choice(bits2))
                    else:
                        new_j = random.randint(0, N-1)

                if new_i == old_i or new_j == old_j:
                    continue
                if new_i == new_j:
                    continue
                if any(new_i == family[k] for k in range(m) if k not in (i, j)):
                    continue
                if any(new_j == family[k] for k in range(m) if k not in (i, j)):
                    continue

                lost = 0
                gained = 0
                for k in range(m):
                    if k == i or k == j:
                        continue
                    other = family[k]
                    if comp[old_i][other] or comp[other][old_i]:
                        lost += 1
                    if comp[old_j][other] or comp[other][old_j]:
                        lost += 1
                    if comp[new_i][other] or comp[other][new_i]:
                        gained += 1
                    if comp[new_j][other] or comp[other][new_j]:
                        gained += 1

                if comp[old_i][old_j] or comp[old_j][old_i]:
                    lost += 1
                if comp[new_i][new_j] or comp[new_j][new_i]:
                    gained += 1

                delta = gained - lost

                if delta > 0:
                    family[i] = new_i
                    family[j] = new_j
                    current_score += delta
                    no_improve = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_family = family[:]
                else:
                    no_improve += 1

            if no_improve > threshold:
                break

        return best_family, best_score

    def construct_star_structure():
        sets = [0]
        for i in range(1, n):
            sets.append(1 | (1 << i))
        candidates_l2 = []
        for i in range(1, n):
            for j in range(i+1, n):
                candidates_l2.append(1 | (1 << i) | (1 << j))
        random.shuffle(candidates_l2)
        sets.extend(candidates_l2[:28])
        return sets

    def construct_chain_structure():
        chain = [0]
        cur = 0
        for b in range(n):
            cur |= (1 << b)
            chain.append(cur)
        remaining = [x for x in range(N) if x not in chain]
        random.seed(42)
        extra = random.sample(remaining, m - len(chain))
        return chain + extra

    best_overall = None
    best_score_overall = 0

    if time.time() < deadline - 15:
        fam0 = construct_star_structure()
        fam, score = local_search(fam0, 150000, deadline)
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam

    if time.time() < deadline - 15:
        fam1 = construct_chain_structure()
        fam, score = local_search(fam1, 150000, deadline)
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam

    for restart in range(2, 7):
        if time.time() > deadline - 15:
            break
        random.seed(100 + restart)
        family = random.sample(range(N), m)
        fam, score = local_search(family, 150000, deadline)
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam

    if best_overall is None:
        best_overall = list(range(m))

    result = []
    for mask in best_overall:
        result.append([b for b in range(n) if (mask >> b) & 1])

    return result
