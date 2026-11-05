import time
import random

def _subset_sum_bitset(elements, k, target):
    """Check if there's a subset of exactly k elements from elements summing to target.
    Uses bitset DP for speed.
    Returns True if such a subset exists."""
    n = len(elements)
    if k < 0 or k > n:
        return False
    if k == 0:
        return target == 0
    if k == n:
        return sum(elements) == target
    if target < 0:
        return False
    
    # Early bounds
    sorted_e = sorted(elements)
    min_s = sum(sorted_e[:k])
    max_s = sum(sorted_e[-k:])
    if target < min_s or target > max_s:
        return False
    
    # Use complement if beneficial
    ck = n - k
    if ck < k:
        ct = sum(elements) - target
        if ct < 0:
            return False
        return _subset_sum_bitset(elements, ck, ct)
    
    if k == 1:
        return target in elements
    if k == 2:
        seen = set()
        for x in elements:
            if (target - x) in seen:
                return True
            seen.add(x)
        return False
    
    # Bitset DP
    # dp[j] is a bitset where bit s is 1 if sum s is achievable with j elements
    # We only care about sums up to target
    max_sum = target
    dp = [0] * (k + 1)
    dp[0] = 1  # bit 0 is set
    
    for e in elements:
        for j in range(k, 0, -1):
            # Shift dp[j-1] by e and mask to max_sum
            # dp[j] |= (dp[j-1] << e)
            # We need to ensure we don't exceed max_sum to keep integers small/fast
            dp[j] |= (dp[j-1] << e)
            # Mask to keep only bits up to max_sum
            # Creating a mask of (max_sum + 1) bits set to 1
            # mask = (1 << (max_sum + 1)) - 1
            # dp[j] &= mask
            # However, creating the mask every time is expensive.
            # Since max_sum <= 512 * 48 ~ 24576, the integer size is manageable.
            # But we can optimize by noting that if e > max_sum, shift is 0 effectively for relevant bits?
            # Actually, simple shift is fine in Python for these sizes.
            
    # Check if bit 'target' is set in dp[k]
    return (dp[k] >> target) & 1 == 1

def _valid_add(S, c):
    """Check if c can be added to set S without violating the mean condition.
    
    Condition 1: c is not the mean of any subset of S of size >= 2
    Condition 2: no x in S is the mean of a subset of (S ∪ {c}) \ {x} of size >= 2 that includes c
    """
    S_list = list(S)
    n = len(S_list)
    
    # Condition 1: c is not the mean of any subset of S of size >= 2
    # sum(subset) = k * c
    for k in range(2, n+1):
        if _subset_sum_bitset(S_list, k, k * c):
            return False
    
    # Condition 2: for each x in S, no subset of (S \ {x}) of size k-1 (k>=2)
    # has sum = k*x - c
    # Note: The subset must include c. So we are looking for a subset T of S \ {x}
    # such that sum(T) + c = k * x => sum(T) = k * x - c.
    # The size of T is k-1.
    for x in S_list:
        others = [y for y in S_list if y != x]
        # k is the size of the full subset including c.
        # The subset of OTHER elements has size k-1.
        # k ranges from 2 to n (since we need at least 2 OTHER elements? 
        # Wait, the condition is "mean of any subset of at least two OTHER elements".
        # If the subset of OTHER elements has size m >= 2, then the full subset including c has size m+1?
        # No.
        # Let's re-read carefully: "No element may be the exact arithmetic mean of any subset of at least two OTHER elements."
        # Let the element be $x$.
        # Let the subset of OTHER elements be $A \subseteq S \setminus \{x\}$ with $|A| \ge 2$.
        # Condition: $x = \frac{\sum_{a \in A} a}{|A|} \implies \sum_{a \in A} a = |A| \cdot x$.
        
        # So when adding $c$ to $S$:
        # 1. $c$ is the new element. Check if $c$ is the mean of any subset $A \subseteq S$ with $|A| \ge 2$.
        #    This is Condition 1 above: $\sum_{a \in A} a = |A| \cdot c$.
        
        # 2. An existing element $x \in S$ might now be the mean of a subset $B$ of $(S \cup \{c\}) \setminus \{x\}$ with $|B| \ge 2$.
        #    The subset $B$ must include $c$? Not necessarily.
        #    If $B$ does not include $c$, then $B \subseteq S \setminus \{x\}$. This condition was already checked when $x$ was added to $S \setminus \{x\}$?
        #    No, when $x$ was added to $S \setminus \{x\}$, the set was smaller.
        #    However, if $B$ does not include $c$, then $B \subseteq S \setminus \{x\}$.
        #    The condition "x is mean of subset of at least two OTHER elements" was checked for all subsets of $S \setminus \{x\}$ when $x$ was added?
        #    Actually, when we added elements to build $S$, we ensured that no element is the mean of a subset of *previously added* elements.
        #    But we didn't check subsets involving *future* elements.
        #    Wait, the greedy algorithm adds elements one by one.
        #    When adding $c$ to $S$, $S$ is the set of previously added elements.
        #    We check if $c$ is the mean of any subset of $S$. (Condition 1)
        #    We check if any $x \in S$ is the mean of any subset of $(S \cup \{c\}) \setminus \{x\}$.
        #    This subset $B$ can either contain $c$ or not.
        #    If $B$ does not contain $c$, then $B \subseteq S \setminus \{x\}$.
        #    Was this checked before?
        #    When $x$ was added to the set (let's call the set at that time $S_{old}$), we checked if $x$ is the mean of any subset of $S_{old}$.
        #    $S_{old} \subset S \setminus \{x\}$.
        #    So subsets of $S \setminus \{x\}$ that were not in $S_{old}$ involve elements added *after* $x$.
        #    So yes, we MUST check all subsets of $(S \cup \{c\}) \setminus \{x\}$ of size $\ge 2$.
        
        # So for each $x \in S$:
        # We need to check if there exists a subset $B \subseteq (S \cup \{c\}) \setminus \{x\}$ with $|B| \ge 2$ such that $\sum_{b \in B} b = |B| \cdot x$.
        
        # Case A: $c \notin B$. Then $B \subseteq S \setminus \{x\}$.
        # Case B: $c \in B$. Then $B = \{c\} \cup A$ where $A \subseteq S \setminus \{x\}$.
        #   Then $\sum_{b \in B} b = c + \sum_{a \in A} a = |B| \cdot x = (|A| + 1) \cdot x$.
        #   So $\sum_{a \in A} a = (|A| + 1) \cdot x - c$.
        #   Let $k = |A| + 1$. Then $|A| = k - 1$.
        #   We need to check if there is a subset $A \subseteq S \setminus \{x\}$ of size $k-1$ summing to $k \cdot x - c$.
        #   Here $k$ is the size of $B$. Since $|B| \ge 2$, and $c \in B$, $|A| \ge 1$.
        #   So $k \ge 2$.
        
        # Case A: $c \notin B$. $B \subseteq S \setminus \{x\}$.
        #   We need to check if there is a subset $B \subseteq S \setminus \{x\}$ of size $m \ge 2$ summing to $m \cdot x$.
        #   This is equivalent to checking if $x$ is the mean of a subset of $S \setminus \{x\}$.
        
        # So for each $x \in S$, we need to check:
        # 1. Is $x$ the mean of any subset of $S \setminus \{x\}$ of size $\ge 2$?
        # 2. Is there a subset $A \subseteq S \setminus \{x\}$ of size $j \ge 1$ such that $\sum_{a \in A} a = (j+1)x - c$?
        
        # Let's implement these checks.
        
        # Check 1: x is mean of subset of S \ {x}
        others = [y for y in S_list if y != x]
        for m in range(2, len(others) + 1):
            if _subset_sum_bitset(others, m, m * x):
                return False
        
        # Check 2: c is in the subset.
        # A subset A of others of size j (j>=1) such that sum(A) = (j+1)*x - c
        for j in range(1, len(others) + 1):
            target_sum = (j + 1) * x - c
            if target_sum < 0:
                continue
            if _subset_sum_bitset(others, j, target_sum):
                return False

    return True

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin before 90s kill
    
    best = []
    best_sz = 0
    
    # Try multiple random orderings with greedy construction
    for trial in range(200):
        if time.time() > deadline - 10:
            break
        
        cands = list(range(1, 513))
        random.shuffle(cands)
        
        cur = []
        for c in cands:
            if time.time() > deadline - 5:
                break
            if _valid_add(cur, c):
                cur.append(c)
        
        if len(cur) > best_sz:
            best_sz = len(cur)
            best = cur[:]
            if best_sz >= 48:
                break
    
    # Local search: try to swap elements to increase size
    if best_sz < 48 and time.time() < deadline - 5:
        cur = best[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(cur)):
                if time.time() > deadline - 5:
                    break
                for c in range(1, 513):
                    if c in cur:
                        continue
                    new_cur = cur[:i] + cur[i+1:]
                    if _valid_add(new_cur, c):
                        cur = new_cur + [c]
                        improved = True
                        break
                if improved:
                    break
        
        if len(cur) > best_sz:
            best_sz = len(cur)
            best = cur[:]
    
    return best[:48]
