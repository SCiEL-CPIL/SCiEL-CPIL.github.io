import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences are distinct across all rows.
    Minimizes the largest entry.
    """
    deadline = time.time() + 140.0  # 10s safety margin for 150s limit
    
    # The number of differences in one row is k * (k+1) / 2
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Lower bound for M: we need total_diffs_needed distinct differences in {1, ..., M}
    # So M >= total_diffs_needed. Also M >= k (trivial).
    # A stronger bound: The sum of the k largest differences in a row with max entry M
    # is at least k*M - k*(k-1)/2 (if the row is {0, 1, ..., k} shifted? No).
    # Actually, for a row a_0=0 < a_1 < ... < a_k=M, the differences involving M are
    # M-a_0, M-a_1, ..., M-a_{k-1}. These are k distinct values in {1, ..., M-1}.
    # The smallest possible sum of these k values is 1+2+...+k = k(k+1)/2.
    # So sum(M-a_i for i<k) >= k(k+1)/2.
    # This doesn't directly bound M tightly without more info.
    
    # Let's use a simple lower bound: M >= total_diffs_needed.
    # For (7,5), total_diffs = 7*15 = 105. So M >= 105.
    # The known best is 112.
    
    # We will search M from lower_bound to some upper_bound.
    # First, get an upper bound via greedy.
    greedy_rows = _greedy_construct(n_rows, k)
    best_M = _max_entry(greedy_rows)
    best_rows = greedy_rows
    
    # Lower bound
    M_lower = max(total_diffs_needed, k + 1)
    
    # We want to find the smallest M such that a solution exists.
    # Iterate M from M_lower to best_M - 1.
    # If we find a solution, update best and stop (since we go upwards, first found is optimal for that M? No, we want global min M).
    # Since we iterate M increasing, the first M for which we find a valid packing is the optimal M (assuming we can prove existence for that M and non-existence for smaller M? No, we just find the smallest M where our solver succeeds. If our solver is complete, it's optimal. If not, it's a heuristic).
    # Given time limits, we treat this as a heuristic search for small M.
    
    # To make it faster, we can parallelize or use a very fast solver for small M.
    # For (7,5), M is ~105-112. The number of candidate rows for M=105 is C(105, 5) ~ 96M. Too many to enumerate.
    # So we must use a smart backtracking or randomized approach.
    
    # Strategy:
    # For each M in range(M_lower, best_M):
    #   Try to solve the packing problem with max entry M.
    #   Use a randomized backtracking with restarts.
    #   If solved, update best_rows and best_M, and break.
    
    # Optimization: If M is small, the search space is tight.
    # We can precompute all valid rows for a given M?
    # For M=105, k=5, C(105,5) is ~96 million. This is too many to store in memory if we store all.
    # But we can generate them on the fly or filter aggressively.
    
    # Alternative: Use the "difference set" formulation.
    # Each row corresponds to a set of 15 differences.
    # We need 7 disjoint sets of differences, each set coming from a valid row structure.
    
    # Let's stick to the row-based backtracking but with a twist:
    # Instead of picking a random row, we pick the row that is "most constrained" or "hardest to fit" later?
    # Or use a priority queue of candidate rows?
    
    # Given the 120s limit, we can afford to try a few M values with a decent solver.
    
    # Let's try M from M_lower to best_M - 1.
    # For each M, run a time-limited solver.
    
    # We will allocate time budget to each M.
    # If M_lower is close to best_M, we have few M values.
    # If M_lower is much smaller, we might not have time to check all.
    # We should prioritize M values close to the known lower bound? No, we want the SMALLEST M.
    # So we must check M_lower first.
    
    # However, if M_lower is infeasible, we move to M_lower+1, etc.
    # This can be slow if many M values are infeasible.
    # But for (7,5), M_lower=105, best_M~115. Only ~10 values to check.
    
    # Let's implement a solver that tries to find a packing for a given M.
    
    for M in range(M_lower, best_M):
        if time.time() > deadline - 5:
            break
            
        # Try to solve for this M
        # Allocate a time slice, e.g., 10 seconds per M?
        # Or just run until we find a solution or time out for this M?
        # If we time out, we assume infeasible (heuristic) and move to next M?
        # This is risky. But given the small range of M, we can give more time to each.
        
        # Let's give each M a budget of (deadline - time.time()) / (best_M - M) * 0.5 ?
        # Simpler: Just try to solve. If it takes too long, we might miss the optimal M.
        # But for (7,5), the optimal M is 112. M_lower is 105.
        # We check 105, 106, ..., 111. If all fail, we keep the greedy result (which might be 115 or 112).
        # Actually, if we find a solution for 112, we stop and return it.
        
        # To be safe, we should not spend all time on M=105 if it's infeasible.
        # Let's use a fixed time limit per M, e.g., 10 seconds.
        # If we have 10 M values, that's 100 seconds. Fits in 140s.
        
        m_deadline = time.time() + 10.0
        if m_deadline > deadline - 5:
            m_deadline = deadline - 5
            
        rows = _solve_packing(n_rows, k, M, m_deadline)
        
        if rows is not None:
            best_rows = rows
            best_M = M
            # Since we iterate M upwards, this is the smallest M found.
            # We can break immediately?
            # Yes, because we want the smallest M.
            break
            
    return best_rows

def _solve_packing(n_rows, k, M, deadline):
    """
    Try to find n_rows rows of k+1 integers from {0, ..., M} starting at 0,
    such that all pairwise differences are distinct.
    Uses randomized backtracking with restarts.
    """
    used_diffs = set()
    rows = []
    
    # To speed up, we can precompute some structures?
    # For a given M, the possible rows are subsets of {1, ..., M} of size k.
    # We can't precompute all if M is large.
    
    # Strategy: Backtrack on rows.
    # At each step, generate a random valid row that doesn't conflict with used_diffs.
    # If we get stuck, backtrack.
    # To improve, we can use "forward checking": when adding a row, we update used_diffs.
    
    # Heuristic for row generation:
    # Instead of purely random, we can bias towards rows that use smaller differences?
    # Or larger?
    # Let's try random first.
    
    def generate_random_row():
        # Pick k distinct values from 1..M
        # Sort them
        # Check if internal differences are distinct
        # Check if global differences are distinct
        vals = random.sample(range(1, M + 1), k)
        vals.sort()
        row = [0] + vals
        
        d_set = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set or d in used_diffs:
                    return None, None
                d_set.add(d)
        return row, d_set
        
    def backtrack(idx):
        if time.time() > deadline:
            return False
            
        if idx == n_rows:
            return True
            
        # Try to find a row
        # To avoid infinite loops on bad choices, we limit attempts
        # and use backtracking.
        
        # We can shuffle the order of attempts?
        # Since we generate random rows, each attempt is different.
        
        # How many attempts?
        # If we fail after many attempts, we backtrack.
        # But backtracking to the previous row and trying a different one there is key.
        
        # Let's try up to 100 random rows for this position.
        # If none work, return False.
        
        for _ in range(100):
            if time.time() > deadline:
                return False
                
            row, d_set = generate_random_row()
            if row is None:
                continue
                
            # Accept
            rows.append(row)
            used_diffs.update(d_set)
            
            if backtrack(idx + 1):
                return True
                
            # Backtrack
            rows.pop()
            used_diffs.difference_update(d_set)
            
        return False
        
    # Multiple restarts to escape local minima
    # Since the search is random, restarts help.
    
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Reset state
        used_diffs.clear()
        rows.clear()
        
        # Seed random for reproducibility in debugging, but here we want diversity
        # random.seed(M * 1000 + r) # Optional
        
        if backtrack(0):
            return rows
            
    return None

def _greedy_construct(n_rows, k):
    """
    Greedy construction: pick the smallest possible next value that doesn't conflict.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used and len(new) == len(row):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _max_entry(rows):
    return max(max(row) for row in rows)

def _is_valid(rows, k):
    """
    Check if rows are valid.
    """
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        if not all(row[i] < row[i+1] for i in range(k)):
            return False
            
        d_set = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set:
                    return False
                if d in used:
                    return False
                d_set.add(d)
        used |= d_set
    return True
