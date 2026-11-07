        # Generate new sums: for each existing sum of cardinality m, add cand to get sum of cardinality m+1
        for m, sum_set in sums_by_cardinality.items():
            new_sums_by_card[m] = set(sum_set)  # copy
            for s in sum_set:
                if m + 1 not in new_sums_by_card:
                    new_sums_by_card[m + 1] = set()
                new_sums_by_card[m + 1].add(s + cand)
        
        # Also add cand itself (cardinality 1)
        if 1 not in new_sums_by_card:
            new_sums_by_card[1] = set()
        new_sums_by_card[1].add(cand)
