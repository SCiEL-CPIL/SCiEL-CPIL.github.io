"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from functools import lru_cache

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all pairwise differences are distinct across all rows.
    Minimize the largest entry.
    """
    # Number of elements per row
    n_elem = k + 1
    # Number of differences per row
    diffs_per_row = n_elem * (n_elem - 1) // 2
    # Total differences needed
    total_diffs = n_rows * diffs_per_row
    
    # We need to find the minimum M such that there exist n_rows valid rows
    # with max element <= M and all differences distinct.
    
    # Lower bound: we need total_diffs distinct positive integers, 
    # so the largest difference is at least total_diffs.
    # But the largest difference in a row is at most M (since a_0=0, a_max<=M).
    # So M >= total_diffs is a lower bound, but actually the differences range from 1 to M,
    # and we need total_diffs of them, so M >= total_diffs.
    
    # Actually, the differences are a_j - a_i where 0 <= i < j <= k, and a_0 = 0, a_k <= M.
    # So each difference is in [1, M]. We need total_diffs distinct values in [1, M].
    # So M >= total_diffs.
    
    # Let's try increasing M from a reasonable lower bound.
    
    # A better lower bound: the sum of all differences in one row is at least 
    # 1+2+...+(n_elem-1) = n_elem*(n_elem-1)/2 if the row is 0,1,2,...,k
    # But that's not quite right. Let's just start from a reasonable value.
    
    # For n_rows=7, k=6: total_diffs = 7*21 = 147
    # So M >= 147? No wait, the differences don't have to be 1..M, they just need to be distinct.
    # The largest difference in a row is at most M (a_k - a_0 = a_k <= M).
    # So all differences are in [1, M], and we need 147 distinct ones, so M >= 147.
    
    # But the known best is 112, so my bound is wrong. Let me reconsider.
    # Ah, I see - the differences are a_j - a_i, and a_j <= M, a_i >= 0, so difference <= M.
    # But we don't need ALL values from 1 to M, just 147 distinct values each <= M.
    # So M must be at least 147? That can't be right since best is 112.
    
    # Wait, I think I'm confusing myself. Let me re-examine.
    # If a row is [0, a1, a2, ..., a6] with a6 <= M, then the differences are:
    # a1-0, a2-0, a3-0, a4-0, a5-0, a6-0,
    # a2-a1, a3-a1, a4-a1, a5-a1, a6-a1,
    # a3-a2, a4-a2, a5-a2, a6-a2,
    # a4-a3, a5-a3, a6-a3,
    # a5-a4, a6-a4,
    # a6-a5
    # All these must be distinct and in [1, M].
    # So we need 21 distinct values in [1, M] per row, 147 total.
    # So M >= 147? But best known is 112...
    
    # I think the issue is that the differences don't all have to be <= M. 
    # Wait, a_j - a_i <= a_j <= M since a_i >= 0. So yes, all differences <= M.
    # So we need 147 distinct integers in [1, M], which requires M >= 147.
    
    # But the problem says best known is 112. Let me re-read the task.
    # "Choose 7 rows of 6 increasing integers starting at 0" - so k=6 means 6 integers? 
    # No, "k + 1 integers" - so k=6 means 7 integers.
    # "all differences between two entries of the same row are distinct over all rows"
    # "the largest entry as small as possible"
    # "The best known set has largest entry 112"
    
    # Hmm, maybe I'm miscounting. Let me check: if we have 7 rows and each row has 7 elements,
    # that's C(7,2)=21 pairs per row, 7*21=147 total pairs. Each pair gives a difference.
    # All 147 differences must be distinct. Each difference is at most M (the max element).
    # So we need 147 distinct positive integers each <= M, so M >= 147.
    
    # But the task says best is 112. There must be something I'm missing.
    # Let me re-read: "7 rows of 6 increasing integers" - maybe it's 6 integers, not 7?
    # The interface says "list of k + 1 integers", and k=6, so 7 integers.
    
    # Wait, maybe the counting bound of 105 is a lower bound on M, not on the number of differences.
    # Let me think differently. Perhaps the differences don't all have to be distinct from 1 to M,
    # but the constraint is just that no two differences (across all rows) are equal.
    # So we have 147 differences, all distinct, all in [1, M]. So M >= 147.
    
    # I'm confused by the 112. Let me just implement a backtracking search and see what M we find.
    # Maybe I'll discover the correct interpretation.
    
    # Let me implement a backtracking search that tries to build rows with a given max bound M.
    
    def can_construct(n_rows, k, M):
        """Check if we can construct n_rows valid rows with max element <= M."""
        n_elem = k + 1
        used_diffs = set()
        rows = []
        
        def backtrack_row(row, pos, used_diffs):
            """Build a row of n_elem elements starting at 0, with current partial row."""
            if pos == n_elem:
                return row[:]
            
            # Try each possible value for the next element
            start = row[-1] + 1 if row else 1
            for val in range(start, M + 1):
                # Check if adding val creates any duplicate differences
                new_diffs = set()
                for prev in row:
                    d = val - prev
                    if d in used_diffs or d in new_diffs:
                        break
                    new_diffs.add(d)
                else:
                    # All differences are new
                    new_used = used_diffs | new_diffs
                    result = backtrack_row(row + [val], pos + 1, new_used)
                    if result is not None:
                        return result
            return None
        
        def backtrack_rows(row_idx, used_diffs, rows):
            """Build all n_rows rows."""
            if row_idx == n_rows:
                return rows[:]
            
            row = backtrack_row([0], 1, used_diffs)
            if row is None:
                return None
            
            # Collect all differences from this row
            row_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    row_diffs.add(row[j] - row[i])
            
            new_used = used_diffs | row_diffs
            result = backtrack_rows(row_idx + 1, new_used, rows + [row])
            if result is not None:
                return result
            return None
        
        return backtrack_rows(0, set(), [])
    
    # Try increasing M
    # Lower bound: we need 147 distinct differences, each in [1, M], so M >= 147
    # But let's start from a reasonable value and increase.
    # The simple counting bound mentioned is 105, so let's try around there.
    
    # Actually, let me reconsider the lower bound. 
    # In each row with elements 0 = a_0 < a_1 < ... < a_k, the differences include a_1, a_2, ..., a_k
    # (differences from a_0=0). So the row contains k distinct positive differences just from the first element.
    # Across n_rows rows, we have n_rows * k distinct differences from the first element alone.
    # For n_rows=7, k=6: 42 distinct differences from the first element, each <= M.
    # Plus other differences. Total 147.
    # So M >= 147 seems right.
    
    # But the task says 112 is best. I must be misunderstanding the problem.
    # Let me just try a range of M values and see.
    
    # Let me try a smarter approach: use the fact that we can search over M.
    
    # For efficiency, let's use a different strategy:
    # Build all rows simultaneously using backtracking, or use a smarter row construction.
    
    # Given time constraints (120s), let's use a practical approach:
    # Try M from a lower bound upward, using backtracking with pruning.
    
    # Let's start with M = 100 and go up.
    
    for M in range(100, 200):
        result = can_construct(n_rows, k, M)
        if result is not None:
            return result
    
    # Fallback: return something valid even if not optimal
    # Use the greedy approach as fallback
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
