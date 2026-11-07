import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 60s limit + safety margin for overhead

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)

    def score_of(values, edge_list):
        sums = set()
        prods = set()
        for u, v in edge_list:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def try_build_edges(values, K, time_limit):
        """
        Backtracking search: find 35 edges such that
        #distinct sums <= K and #distinct products <= K.
        Returns (edges, sums_used, prods_used) or None.
        """
        # Precompute sum and product for every edge
        edge_sum = {}
        edge_prod = {}
        for e in all_edges:
            u, v = e
            edge_sum[e] = values[u] + values[v]
            edge_prod[e] = values[u] * values[v]

        # All possible sums and products that appear on any edge
        all_sums = sorted(set(edge_sum.values()))
        all_prods = sorted(set(edge_prod.values()))

        # For each edge, record which sum-index and prod-index it uses
        sum_index = {s: i for i, s in enumerate(all_sums)}
        prod_index = {p: i for i, p in enumerate(all_prods)}

        edge_si = {}
        edge_pi = {}
        for e in all_edges:
            edge_si[e] = sum_index[edge_sum[e]]
            edge_pi[e] = prod_index[edge_prod[e]]

        # For a given K, we can only use sums/prods that are "available".
        # Strategy: choose a subset of K sums and K prods, then see if 35 edges fit.
        # This is combinatorially hard, so use a heuristic:
        #   - Start with the most "popular" sums and prods (those appearing on most edges)
        #   - Build allowed edge set
        #   - Use backtracking / greedy to pick 35 edges

        # Count frequency of each sum and product
        sum_freq = {}
        prod_freq = {}
        for e in all_edges:
            s = edge_sum[e]
            p = edge_prod[e]
            sum_freq[s] = sum_freq.get(s, 0) + 1
            prod_freq[p] = prod_freq.get(p, 0) + 1

        # Sort sums and prods by frequency (descending)
        top_sums = sorted(sum_freq.keys(), key=lambda x: -sum_freq[x])
        top_prods = sorted(prod_freq.keys(), key=lambda x: -prod_freq[x])

        # Try several choices of K sums and K prods
        # Candidate sum-sets: top K, top K-1 + next, etc.
        # For simplicity, start with top K and try variations

        best_result = None

        # Number of variations to try
        n_variations = 8

        for var in range(n_variations):
            if time.time() > time_limit:
                break

            # Choose K sums and K prods
            if var == 0:
                chosen_sums = set(top_sums[:K])
                chosen_prods = set(top_prods[:K])
            else:
                # Randomly replace some
                chosen_sums = set(top_sums[:K])
                chosen_prods = set(top_prods[:K])
                for _ in range(var):
                    if chosen_sums:
                        to_remove = random.choice(list(chosen_sums))
                        chosen_sums.discard(to_remove)
                        candidates = [s for s in top_sums if s not in chosen_sums]
                        if candidates:
                            chosen_sums.add(random.choice(candidates))
                    if chosen_prods:
                        to_remove = random.choice(list(chosen_prods))
                        chosen_prods.discard(to_remove)
                        candidates = [p for p in top_prods if p not in chosen_prods]
                        if candidates:
                            chosen_prods.add(random.choice(candidates))

            # Allowed edges: those whose sum is in chosen_sums AND product in chosen_prods
            allowed = [e for e in all_edges if edge_sum[e] in chosen_sums and edge_prod[e] in chosen_prods]

            if len(allowed) < m:
                continue

            # Now pick 35 edges from allowed such that we don't exceed K distinct sums and K distinct prods.
            # But since all allowed edges already use only sums in chosen_sums (size K) and prods in chosen_prods (size K),
            # any 35 edges from allowed will satisfy the constraint.
            # So just pick any 35.
            random.shuffle(allowed)
            edges = allowed[:m]

            # Verify
            sums_used = set(edge_sum[e] for e in edges)
            prods_used = set(edge_prod[e] for e in edges)
            if len(sums_used) <= K and len(prods_used) <= K:
                result = (edges, sums_used, prods_used)
                if best_result is None or len(sums_used) + len(prods_used) < len(best_result[1]) + len(best_result[2]):
                    best_result = result
                    if len(sums_used) == K and len(prods_used) == K:
                        break  # Can't do better than K,K

        return best_result

    def try_build_edges_with_backtrack(values, K, time_limit):
        """
        More careful backtracking: build edges one by one, tracking which sums/prods are used.
        """
        edge_sum = {}
        edge_prod = {}
        for e in all_edges:
            u, v = e
            edge_sum[e] = values[u] + values[v]
            edge_prod[e] = values[u] * values[v]

        all_sums = sorted(set(edge_sum.values()))
        all_prods = sorted(set(edge_prod.values()))
        sum_index = {s: i for i, s in enumerate(all_sums)}
        prod_index = {p: i for i, p in enumerate(all_prods)}

        # Frequency
        sum_freq = {}
        prod_freq = {}
        for e in all_edges:
            sum_freq[edge_sum[e]] = sum_freq.get(edge_sum[e], 0) + 1
            prod_freq[edge_prod[e]] = prod_freq.get(edge_prod[e], 0) + 1

        top_sums = sorted(sum_freq.keys(), key=lambda x: -sum_freq[x])
        top_prods = sorted(prod_freq.keys(), key=lambda x: -prod_freq[x])

        # Try a few initial choices of K sums and K prods
        best_result = None

        for var in range(5):
            if time.time() > time_limit:
                break

            if var == 0:
                chosen_sums = set(top_sums[:K])
                chosen_prods = set(top_prods[:K])
            else:
                chosen_sums = set(top_sums[:K])
                chosen_prods = set(top_prods[:K])
                for _ in range(var):
                    if chosen_sums:
                        to_remove = random.choice(list(chosen_sums))
                        chosen_sums.discard(to_remove)
                        candidates = [s for s in top_sums if s not in chosen_sums]
                        if candidates:
                            chosen_sums.add(random.choice(candidates))
                    if chosen_prods:
                        to_remove = random.choice(list(chosen_prods))
                        chosen_prods.discard(to_remove)
                        candidates = [p for p in top_prods if p not in chosen_prods]
                        if candidates:
                            chosen_prods.add(random.choice(candidates))

            # Allowed edges
            allowed = [e for e in all_edges if edge_sum[e] in chosen_sums and edge_prod[e] in chosen_prods]

            if len(allowed) < m:
                continue

            # Greedy: pick edges that minimize new distinct sums/prods
            random.shuffle(allowed)
            edges = []
            sums_used = set()
            prods_used = set()
            edge_set = set()

            # Sort allowed by "impact": prefer edges whose sum and product are already in used sets
            def impact(e):
                s = edge_sum[e]
                p = edge_prod[e]
                return (0 if s in sums_used else 1, 0 if p in prods_used else 1)

            # Iterative greedy
            remaining = list(allowed)
            while len(edges) < m and remaining:
                # Pick the edge with minimum impact
                best_idx = 0
                best_imp = impact(remaining[0])
                for i in range(1, len(remaining)):
                    imp = impact(remaining[i])
                    if imp < best_imp:
                        best_imp = imp
                        best_idx = i
                e = remaining.pop(best_idx)
                edges.append(e)
                edge_set.add(e)
                sums_used.add(edge_sum[e])
                prods_used.add(edge_prod[e])

            if len(edges) == m:
                result = (edges, sums_used, prods_used)
                if best_result is None or (len(sums_used), len(prods_used)) < (len(best_result[1]), len(best_result[2])):
                    best_result = result

        return best_result

    # Candidate value sets designed for collision-friendly sums and products
    candidate_value_sets = [
        # Products of small primes
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16],
        # Powers of 2 mixed with small multiples
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24],
        # Highly composite-like
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        # Small range
        list(range(1, 15)),
        # Mixed
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 18, 20],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18],
    ]

    best_score = float('inf')
    best_vals = None
    best_edges = None

    # For each candidate value set, try to build edges with increasing K
    for vals in candidate_value_sets:
        if time.time() > deadline - 5.0:
            break

        # Ensure distinct and in range
        if len(set(vals)) != n or any(v < 1 or v > 10000 for v in vals):
            continue

        # Try K from a low value upward
        # Minimum possible K: at least ceil(m / max_edges_per_sum) etc.
        # Just try K = 5, 6, 7, 8, 9, 10, 11, 12
        for K in range(5, 13):
            if time.time() > deadline - 5.0:
                break

            result = try_build_edges_with_backtrack(vals, K, deadline - 5.0)
            if result is not None:
                edges, sums_used, prods_used = result
                score = max(len(sums_used), len(prods_used))
                if score < best_score:
                    best_score = score
                    best_vals = list(vals)
                    best_edges = list(edges)
                    # If we achieved K,K, that's optimal for this K
                    if score == K:
                        break

            # Also try the simpler method
            if time.time() < deadline - 5.0:
                result2 = try_build_edges(vals, K, deadline - 5.0)
                if result2 is not None:
                    edges, sums_used, prods_used = result2
                    score = max(len(sums_used), len(prods_used))
                    if score < best_score:
                        best_score = score
                        best_vals = list(vals)
                        best_edges = list(edges)
                        if score == K:
                            break

    # If no good result, fall back to local search on best candidate
    if best_vals is None or best_score > 10:
        # Use the first candidate and do local search
        vals = candidate_value_sets[0]
        # Greedy edge selection
        edge_sum = {}
        edge_prod = {}
        for e in all_edges:
            u, v = e
            edge_sum[e] = vals[u] + vals[v]
            edge_prod[e] = vals[u] * vals[v]

        # Sort by product
        sorted_edges = sorted(all_edges, key=lambda e: (edge_prod[e], edge_sum[e]))
        edges = sorted_edges[:m]
        score = score_of(vals, edges)

        if score < best_score:
            best_score = score
            best_vals = vals
            best_edges = edges

    # Final local search polish on edges
    if best_vals is not None and time.time() < deadline - 3.0:
        vals = best_vals
        edges = list(best_edges)
        edges_set = set(edges)
        cur_score = score_of(vals, edges)

        edge_sum = {}
        edge_prod = {}
        for e in all_edges:
            u, v = e
            edge_sum[e] = vals[u] + vals[v]
            edge_prod[e] = vals[u] * vals[v]

        no_improve = 0
        it = 0
        max_iters = 50000
        while it < max_iters and time.time() < deadline - 2.0:
            it += 1
            i = random.randint(0, m - 1)
            e_out = edges[i]
            e_in = random.choice(all_edges)
            if e_in in edges_set:
                continue

            new_edges = list(edges)
            new_edges[i] = e_in
            new_score = score_of(vals, new_edges)

            if new_score < cur_score:
                edges = new_edges
                edges_set.discard(e_out)
                edges_set.add(e_in)
                cur_score = new_score
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 20000 and random.random() < 0.02:
                    no_improve = 0

        if cur_score < best_score:
            best_score = cur_score
            best_edges = edges

    # Ensure valid output
    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])

    # Validate
    if len(set(best_vals)) != n:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])

    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]

    return {'values': final_values, 'edges': final_edges}
