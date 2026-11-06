"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s limit
    n = 512
    
    def _subset_sum_bitset(arr, max_sum):
        """Return list of bitsets where bitsets[k] has bit s set if some subset of arr of size k sums to s."""
        m = len(arr)
        if m == 0:
            return [1]  # only empty subset sum 0
        # dp[k] is a bitset (Python int) where bit s is set if a subset of size k sums to s
        dp = [0] * (m + 1)
        dp[0] = 1  # sum 0 with size 0
        for val in arr:
            # update in reverse to avoid using same element twice
            for k in range(m, 0, -1):
                # shift dp[k-1] left by val, mask to max_sum
                shifted = (dp[k-1] << val) & ((1 << (max_sum + 1)) - 1)
                dp[k] |= shifted
        return dp
    
    def _can_form_sum(arr, target_sum, min_size, max_size):
        """Check if any subset of arr with size in [min_size, max_size] sums to target_sum."""
        if target_sum < 0 or target_sum > sum(arr):
            return False
        m = len(arr)
        if min_size > m:
            return False
        max_size = min(max_size, m)
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in arr:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << val)
        for k in range(min_size, max_size + 1):
            if (dp[k] >> target_sum) & 1:
                return True
        return False
    
    def _is_valid_incremental(existing, x):
        """Check if adding x to existing set keeps it valid."""
        if not existing:
            return True
        # Check 1: x is not the mean of any subset of existing of size >= 2
        for k in range(2, len(existing) + 1):
            target = k * x
            if target > sum(existing):
                break
            if _can_form_sum(existing, target, k, k):
                return False
        # Check 2: no element a in existing is the mean of a subset of existing∪{x} of size >= 2 that includes x
        # i.e., for some a in existing, there exists subset T of existing, |T| >= 1, such that
        # (sum(T) + x) / (|T| + 1) = a  =>  sum(T) = (|T|+1)*a - x
        for a in existing:
            # |T|+1 >= 2 => |T| >= 1, and |T| <= len(existing)
            for t_size in range(1, len(existing) + 1):
                target = (t_size + 1) * a - x
                if target < 0:
                    continue
                if target > sum(existing):
                    break
                if _can_form_sum(existing, target, t_size, t_size):
                    return False
        return True
    
    selected = []
    for x in range(1, n + 1):
        if time.time() > deadline - 5.0:
            break
        if _is_valid_incremental(selected, x):
            selected.append(x)
            if len(selected) >= 48:
                break
    
    return selected
# EVOLVE-BLOCK-END
