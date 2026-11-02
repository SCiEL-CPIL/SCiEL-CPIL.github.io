"""Improved covering design constructor with time-bounded iterative greedy."""
import itertools
import time


def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin before 120s kill
    all_points = set(range(v))
    # Precompute all t-subsets for membership testing
    all_t_subs = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subs)
    blocks = []

    if not uncovered:
        return blocks

    # Helper: for a given partial block (list of points), compute the gain
    # of adding point x: number of uncovered t-subsets that would be covered
    # by x combined with existing block points.
    def compute_gains(block, uncovered_set):
        """Return dict mapping point -> gain if point not in block."""
        gains = {}
        # We need to count, for each candidate x not in block, how many
        # uncovered t-subsets S contain x and (t-1) points from block.
        # A t-subset S is covered by adding x to block iff S ⊆ block ∪ {x}.
        # That means S contains x and S ∩ block has size t-1, i.e.,
        # S \ {x} ⊆ block and |S \ {x}| = t-1.
        # So S = {x} ∪ R where R ⊆ block, |R| = t-1, and S is uncovered.
        if len(block) < t - 1:
            # Can't form any t-subset yet; all gains are 0 (or we just return empty)
            return {x: 0 for x in all_points if x not in block}
        # For each (t-1)-subset R of block, for each uncovered S containing x and R...
        # More efficient: for each uncovered S, if |S ∩ block| >= t-1, then any x in S \ block
        # could potentially complete it. But we need exactly |S ∩ block| == t-1 and x ∈ S.
        # Actually: S is covered by block ∪ {x} iff S ⊆ block ∪ {x}, i.e., S \ {x} ⊆ block.
        # So for a fixed x, gain(x) = #{S ∈ uncovered : x ∈ S and S \ {x} ⊆ block}
        #                           = #{S ∈ uncovered : x ∈ S and S ∩ block has size t-1 and x ∉ block}
        # Since x ∉ block, S ∩ block has size t-1 means exactly t-1 elements of S are in block.
        # Let's iterate over uncovered sets and distribute.
        # But that's expensive. Alternative: for each (t-1)-subset R of block,
        # the t-subsets that use R and one more point x are R ∪ {x} for x ∉ R.
        # So gain(x) = #{R ⊆ block, |R|=t-1 : R ∪ {x} ∈ uncovered and x ∉ R}
        # Since x ∉ block and R ⊆ block, x ∉ R automatically.
        # So gain(x) = #{R ⊆ block, |R|=t-1 : R ∪ {x} ∈ uncovered}
        # We can precompute for each (t-1)-subset R of block, which uncovered sets are R ∪ {x}.
        # For efficiency, iterate over uncovered sets:
        # For each S in uncovered, if |S ∩ block| == t-1, then for each x in S \ block,
        # gain[x] += 1. But |S \ block| = 1 in this case (since |S|=t and |S∩block|=t-1).
        # So for each S in uncovered with |S ∩ block| == t-1, let x = the unique element of S not in block.
        # Then gain[x] += 1.
        # This is O(|uncovered| * t) per block construction step, which might be slow.
        # Alternative: iterate over (t-1)-subsets of block. There are C(len(block), t-1) of them.
        # For each R, for each x in all_points \ block, check if R ∪ {x} in uncovered.
        # That's C(len(block), t-1) * (v - len(block)) checks.
        # For v=19, k=8, t=5: len(block) up to 7, C(7,4)=35, v-len=12, so 420 checks. Fine.
        block_set = set(block)
        for R in itertools.combinations(block, t - 1):
            R_set = set(R)
            for x in all_points:
                if x in R_set:
                    continue
                S = frozenset(R_set | {x})
                # Check if S is in uncovered
                if tuple(sorted(S)) in uncovered_set:
                    gains[x] = gains.get(x, 0) + 1
        # Fill in zeros for points not in gains
        for x in all_points:
            if x not in block_set and x not in gains:
                gains[x] = 0
        return gains

    while uncovered and time.time() < deadline:
        # Start a new block with the smallest uncovered t-subset
        first_uncovered = min(uncovered)
        block = list(first_uncovered)
        block_set = set(block)
        
        while len(block) < k and time.time() < deadline:
            gains = compute_gains(block, uncovered)
            # Pick the point with maximum gain; break ties by smallest point
            best_x = min(gains.keys(), key=lambda x: (-gains[x], x))
            if gains[best_x] == 0:
                # No gain from any point; pick the smallest unused point to fill
                best_x = min(all_points - block_set)
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Remove all t-subsets of this block from uncovered
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
    
    # Return as list of lists
    return [list(b) for b in blocks]
