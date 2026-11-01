# Allocate budget with more weight to smaller ranks
weights = [max(1.0, 3.0 - 0.1 * idx) for idx in range(len(candidate_ranks))]
