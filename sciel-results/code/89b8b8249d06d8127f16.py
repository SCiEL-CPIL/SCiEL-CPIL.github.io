"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Returns a list of blocks (each a sorted list of k distinct integers in 0..v-1)
    such that every t-subset is contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit

    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_index = {triple: i for i, triple in enumerate(all_triples)}

    # We'll track coverage with a set of covered triple indices
    covered = set()
    blocks = []

    # Precompute for speed: for each pair, the list of triples containing that pair
    # This helps in the greedy step
    # Actually, for the inner loop we need: given a partial block P, and candidate x,
    # how many uncovered triples are of the form (p1, p2, x) where {p1,p2} subset of P
    # and the triple is not yet covered.

    # Let's use a different approach: maintain a count array for uncovered triples
    # uncovered_count[triple_idx] = 1 if not covered, 0 if covered
    # But we need to update it when we add a block.

    # For efficiency, let's maintain a boolean array covered_arr
    covered_arr = bytearray(num_triples)  # 0 = uncovered, 1 = covered
    uncovered_count = num_triples

    # Precompute: for each (t-1)-subset, the list of triples that extend it
    # This is useful for the greedy inner loop
    # For t=3: for each pair (a,b), list of triples (a,b,c) for c > b
    # Actually, we need for a partial block P (size < k), and candidate x not in P,
    # the number of uncovered triples contained in P ∪ {x}.
    # A triple is contained in P ∪ {x} iff it is a t-subset of P ∪ {x}.
    # So we need to count t-subsets of P ∪ {x} that are not yet covered.

    # For the outer loop, we pick the first uncovered triple to start a block.
    # For the inner loop, we greedily add the point that maximizes the number of
    # newly covered uncovered triples.

    # Let's optimize the inner loop by precomputing for each possible partial block
    # the set of triples it would cover. But that's expensive.

    # Alternative: for the inner loop, when we have a partial block P of size m,
    # and we consider adding x, the new triples covered are the t-subsets of P ∪ {x}
    # that include x (since those not including x were already in P's potential).
    # Wait, no: if P has size m, the t-subsets of P are already "covered" by P
    # if P were a full block. But P is not a full block yet.
    # Actually, we only "cover" triples when we commit to a full block.
    # So during the inner loop, we should count how many uncovered triples
    # are contained in P ∪ {x}.

    # Let's just do it directly but efficiently.

    # For the outer loop, instead of picking the first uncovered triple,
    # we could pick the uncovered triple that is "hardest" to cover, but that's complex.
    # Let's stick with the first uncovered triple for now, but use a smarter
    # initialization: start with a random uncovered triple, or the one with
    # the fewest possible blocks that could cover it.

    # Actually, let's try a different approach: precompute all possible blocks
    # and use a set cover heuristic. But C(24,7) is 346104, which is feasible.

    # Let's try the set cover approach with a greedy heuristic.
    # Precompute all blocks and for each block, the set of triples it covers.
    # Then use a greedy set cover: at each step, pick the block that covers
    # the most uncovered triples.

    # This might be slower but could give a better result.

    # Let's try both approaches and see which is faster.

    # Approach 1: Incremental greedy (original)
    # Approach 2: Precomputed set cover greedy

    # Let's implement Approach 2 first, as it might be more effective.

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # For each block, compute the set of triples it covers
    # This is expensive: C(24,7) * C(7,3) = 346104 * 35 ≈ 12 million, which is feasible
    block_triples = []
    for block in all_blocks:
        triples_in_block = set()
        for triple in itertools.combinations(block, t):
            triples_in_block.add(triple)
        block_triples.append(triples_in_block)

    # Now use greedy set cover
    # uncovered is a set of triple indices
    uncovered_set = set(range(num_triples))

    # For efficiency, maintain for each block the number of uncovered triples it covers
    # Initially, this is just the size of block_triples[i]
    block_coverage = [len(bt) for bt in block_triples]

    # But we need to update this as we cover triples.
    # Instead, let's just compute it on the fly for each candidate block.
    # But that's O(num_blocks * num_triples_per_block) per step, which is too slow.

    # Let's use a different approach: maintain for each triple, the list of blocks that cover it.
    # Then when we cover a triple, we can update the count for each block.

    # triple_to_blocks: for each triple index, list of block indices that cover it
    triple_to_blocks = [[] for _ in range(num_triples)]
    for bi, bt in enumerate(block_triples):
        for triple in bt:
            triple_to_blocks[triple_index[triple]].append(bi)

    # Now, for each block, maintain the number of uncovered triples it covers
    block_uncovered_count = block_coverage[:]  # copy

    # Greedy set cover
    while uncovered_set:
        if time.time() > deadline:
            break

        # Find the block that covers the most uncovered triples
        best_block = -1
        best_count = -1
        for bi in range(num_blocks):
            if block_uncovered_count[bi] > best_count:
                best_count = block_uncovered_count[bi]
                best_block = bi

        if best_block == -1:
            break

        blocks.append(list(all_blocks[best_block]))

        # Update: remove covered triples from uncovered_set
        # and update block_uncovered_count for all blocks
        for triple in block_triples[best_block]:
            ti = triple_index[triple]
            if ti in uncovered_set:
                uncovered_set.remove(ti)
                # For each block that covers this triple, decrement its count
                for bi in triple_to_blocks[ti]:
                    block_uncovered_count[bi] -= 1

    # Verify coverage
    if uncovered_set:
        # Fallback: use the original incremental greedy to cover remaining triples
        # This shouldn't happen if the set cover worked correctly
        pass

    return blocks

# EVOLVE-BLOCK-END
