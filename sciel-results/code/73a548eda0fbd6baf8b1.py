import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin for 120s limit

    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)  # 560

    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute all K(2,2,2) configurations
    k222_list = []
    triple_to_k222s = [[] for _ in range(num_triples)]

    for points in combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = points
        pairings = [
            ((p0,p1),(p2,p3),(p4,p5)),
            ((p0,p1),(p2,p4),(p3,p5)),
            ((p0,p1),(p2,p5),(p3,p4)),
            ((p0,p2),(p1,p3),(p4,p5)),
            ((p0,p2),(p1,p4),(p3,p5)),
            ((p0,p2),(p1,p5),(p3,p4)),
            ((p0,p3),(p1,p2),(p4,p5)),
            ((p0,p3),(p1,p4),(p2,p5)),
            ((p0,p3),(p1,p5),(p2,p4)),
            ((p0,p4),(p1,p2),(p3,p5)),
            ((p0,p4),(p1,p3),(p2,p5)),
            ((p0,p4),(p1,p5),(p2,p3)),
            ((p0,p5),(p1,p2),(p3,p4)),
            ((p0,p5),(p1,p3),(p2,p4)),
            ((p0,p5),(p1,p4),(p2,p3)),
        ]
        for pair1, pair2, pair3 in pairings:
            transversals = []
            for x in pair1:
                for y in pair2:
                    for z in pair3:
                        transversals.append(tuple(sorted((x, y, z))))
            trans_indices = [triple_to_idx[t] for t in transversals]
            k_idx = len(k222_list)
            k222_list.append(set(trans_indices))
            for idx in trans_indices:
                triple_to_k222s[idx].append(k_idx)

    num_k222s = len(k222_list)

    # Precompute conflict lists: for each triple, the set of triples that share a K(2,2,2)
    triple_conflicts = [set() for _ in range(num_triples)]
    for k_idx in range(num_k222s):
        members = list(k222_list[k_idx])
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                triple_conflicts[members[i]].add(members[j])
                triple_conflicts[members[j]].add(members[i])

    best_result = []
    best_count = 0

    def greedy_pass(order, time_budget):
        """Greedy add triples in given order, skipping those that would complete a K(2,2,2)."""
        k222_count = [0] * num_k222s
        present = set()
        for triple_idx in order:
            if time.time() > time_budget:
                break
            can_add = True
            for k_idx in triple_to_k222s[triple_idx]:
                if k222_count[k_idx] == 7:
                    can_add = False
                    break
            if can_add:
                present.add(triple_idx)
                for k_idx in triple_to_k222s[triple_idx]:
                    k222_count[k_idx] += 1
        return present

    def polish(present, time_budget):
        """Local search: try removing a triple to allow adding more."""
        present = set(present)
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            k222_count = [0] * num_k222s
            for t in present:
                for k_idx in triple_to_k222s[t]:
                    k222_count[k_idx] += 1

            # Find triples to remove: those in K(2,2,2)s with count >= 6
            # Removing one from a K(2,2,2) with count 7 frees it
            candidates_to_remove = []
            for t in present:
                for k_idx in triple_to_k222s[t]:
                    if k222_count[k_idx] == 7:
                        candidates_to_remove.append(t)
                        break

            if not candidates_to_remove:
                break

            random.shuffle(candidates_to_remove)
            for rem in candidates_to_remove:
                if time.time() > time_budget:
                    break
                # Try removing rem
                present.remove(rem)
                for k_idx in triple_to_k222s[rem]:
                    k222_count[k_idx] -= 1

                # Try to add as many as possible
                added = 0
                for t in range(num_triples):
                    if t in present:
                        continue
                    can_add = True
                    for k_idx in triple_to_k222s[t]:
                        if k222_count[k_idx] == 7:
                            can_add = False
                            break
                    if can_add:
                        present.add(t)
                        added += 1
                        for k_idx in triple_to_k222s[t]:
                            k222_count[k_idx] += 1

                if added > 1:
                    improved = True
                else:
                    # Revert
                    present.difference_update(
                        [t for t in present if t != rem]
                    )
                    # Re-add rem
                    present.add(rem)
                    for k_idx in triple_to_k222s[rem]:
                        k222_count[k_idx] += 1
                    # Remove the ones we added
                    # Actually this revert is complex; just skip
                    pass

        return present

    # Phase 1: Multiple greedy passes with different orderings
    num_greedy = 30
    for g in range(num_greedy):
        if time.time() > deadline - 20:
            break
        order = list(range(num_triples))
        random.shuffle(order)
        present = greedy_pass(order, deadline - 20)
        if len(present) > best_count:
            best_count = len(present)
            best_result = [list(all_triples[i]) for i in present]

    # Phase 2: Greedy with dynamic scoring (prefer triples with fewer conflicts among remaining)
    if time.time() < deadline - 15:
        for g in range(10):
            if time.time() > deadline - 15:
                break
            remaining = set(range(num_triples))
            k222_count = [0] * num_k222s
            present = set()
            while remaining and time.time() < deadline - 15:
                # Score each remaining triple: prefer fewer conflicts with other remaining
                best_t = -1
                best_score = -1
                sample = list(remaining)
                random.shuffle(sample)
                for t in sample[:80]:  # sample for speed
                    conflicts_in_remaining = len(triple_conflicts[t] & remaining)
                    # Also prefer triples in fewer K(2,2,2)s
                    score = -conflicts_in_remaining + random.random()
                    if score > best_score:
                        best_score = score
                        best_t = t
                if best_t < 0:
                    break
                can_add = True
                for k_idx in triple_to_k222s[best_t]:
                    if k222_count[k_idx] == 7:
                        can_add = False
                        break
                if can_add:
                    present.add(best_t)
                    remaining.discard(best_t)
                    for k_idx in triple_to_k222s[best_t]:
                        k222_count[k_idx] += 1
                else:
                    remaining.discard(best_t)
            if len(present) > best_count:
                best_count = len(present)
                best_result = [list(all_triples[i]) for i in present]

    # Phase 3: Polish the best result
    if best_count > 0 and time.time() < deadline - 5:
        best_present = set(i for i in range(num_triples)
                          if tuple(sorted(all_triples[i])) in set(best_result))
        # Rebuild present set from best_result
        best_set = set(tuple(sorted(t)) for t in best_result)
        best_present = set(triple_to_idx[t] for t in best_set)
        polished = polish(best_present, deadline - 3)
        if len(polished) > best_count:
            best_count = len(polished)
            best_result = [list(all_triples[i]) for i in polished]

    # Fallback
    if best_count == 0:
        k222_count = [0] * num_k222s
        present = set()
        for t in range(num_triples):
            can_add = True
            for k_idx in triple_to_k222s[t]:
                if k222_count[k_idx] == 7:
                    can_add = False
                    break
            if can_add:
                present.add(t)
                for k_idx in triple_to_k222s[t]:
                    k222_count[k_idx] += 1
        best_result = [list(all_triples[i]) for i in present]

    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Number of triples: {len(result)}")
