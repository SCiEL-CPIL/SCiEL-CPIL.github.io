import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    N = 512
    MAXC = 48

    def build_dp(S):
        """
        Builds DP bitsets for subset sums of S.
        dp[k] is a bitset where bit s is 1 if s is the sum of exactly k elements from S.
        """
        n = len(S)
        dp = [0] * (n + 1)
        dp[0] = 1  # Sum 0 with 0 elements
        for a in S:
            # Update in reverse to avoid using the same element multiple times in one step
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k + 1] |= dp[k] << a
        return dp

    def verify(S):
        """
        Verifies the entire set S.
        No element in S is the mean of any subset of at least 2 OTHER elements.
        """
        S = list(S)
        n = len(S)
        for i in range(n):
            a = S[i]
            sub = S[:i] + S[i + 1:]
            m = len(sub)
            if m < 2:
                continue
            dp = build_dp(sub)
            maxs = sum(sub)
            for k in range(2, m + 1):
                t = k * a
                if t > maxs:
                    break
                if (dp[k] >> t) & 1:
                    return False
        return True

    # Precompute pair_conflict: for each x, count pairs (u,v) in [1,N]\{x} with u+v=2x
    pair_conflict = [0] * (N + 1)
    for x in range(1, N + 1):
        for u in range(1, x):
            v = 2 * x - u
            if v > x and v <= N:
                pair_conflict[x] += 1

    # Precompute 3-AP degrees
    deg = [0] * (N + 1)
    for a in range(1, N - 1):
        for c in range(a + 2, N + 1, 2):
            b = (a + c) // 2
            if 1 <= b <= N:
                deg[a] += 1
                deg[b] += 1
                deg[c] += 1

    best = []
    best_sz = 0

    class DPContext:
        """Maintains DP state for a growing set, with efficient add/remove."""
        def __init__(self):
            self.S = []
            self.n = 0
            self.maxsum = 0
            self.dp_full = [1]  # dp[0]=1
            self.dp_without = {}  # element -> (dp_bitsets, maxsum_without)

        def add(self, x):
            """Add element x to the set and update all DP structures."""
            self.n += 1
            old_n = self.n - 1
            old_maxsum = self.maxsum

            # Build new dp_full from old
            new_dp = [0] * (self.n + 1)
            new_dp[0] = 1
            for a in self.S:
                for k in range(old_n, -1, -1):
                    if new_dp[k]:
                        new_dp[k + 1] |= new_dp[k] << a
            for k in range(old_n + 1, -1, -1):
                if new_dp[k]:
                    new_dp[k + 1] |= new_dp[k] << x

            # Update dp_without: for each existing element a, recompute dp without a
            # Also add dp_without[x]
            new_dp_without = {}
            for a in self.S:
                sub = [y for y in self.S if y != a]
                new_dp_without[a] = (build_dp(sub), old_maxsum - a)
            sub_x = self.S[:]  # S without x is just self.S
            new_dp_without[x] = (build_dp(sub_x), old_maxsum)

            self.S.append(x)
            self.maxsum = old_maxsum + x
            self.dp_full = new_dp
            self.dp_without = new_dp_without

        def remove(self, x):
            """Remove element x from the set and update all DP structures."""
            if x not in self.S:
                return
            idx = self.S.index(x)
            self.S.pop(idx)
            self.n -= 1
            self.maxsum -= x

            # Rebuild dp_full
            new_dp = [0] * (self.n + 1)
            new_dp[0] = 1
            for a in self.S:
                for k in range(self.n - 1, -1, -1):
                    if new_dp[k]:
                        new_dp[k + 1] |= new_dp[k] << a

            # Rebuild dp_without
            new_dp_without = {}
            for a in self.S:
                sub = [y for y in self.S if y != a]
                new_dp_without[a] = (build_dp(sub), self.maxsum - a)

            self.dp_full = new_dp
            self.dp_without = new_dp_without

        def can_add(self, x):
            """Check if x can be added to current set without violating constraints."""
            n = self.n
            maxfull = self.maxsum

            # Check 1: x is not the mean of any subset of S (size >= 2)
            for k in range(2, n + 1):
                t = k * x
                if t > maxfull:
                    break
                if (self.dp_full[k] >> t) & 1:
                    return False

            # Check 2: no a in S becomes mean of a subset containing x
            for a in self.S:
                dpo, maxo = self.dp_without[a]
                m = n - 1  # size of S \ {a}
                for j in range(1, m + 1):
                    t = (j + 1) * a - x
                    if t < 0 or t > maxo:
                        continue
                    if (dpo[j] >> t) & 1:
                        return False
            return True

    def greedy_build(order, ctx=None):
        """Greedily build a set following the given order."""
        if ctx is None:
            ctx = DPContext()
        Sset = set(ctx.S)
        for x in order:
            if len(ctx.S) >= MAXC:
                break
            if time.time() > deadline - 5:
                break
            if x in Sset:
                continue
            if ctx.can_add(x):
                ctx.add(x)
                Sset.add(x)
        return ctx

    def local_search(ctx):
        """Improve set via additions and swaps."""
        Sset = set(ctx.S)
        unused = [x for x in range(1, N + 1) if x not in Sset]
        unused.sort(key=lambda x: pair_conflict[x])

        improved = True
        no_improve = 0

        while improved and time.time() < deadline - 3:
            improved = False

            # Try to add an unused element
            added = False
            for u in unused:
                if len(ctx.S) >= MAXC:
                    break
                if time.time() > deadline - 3:
                    break
                if ctx.can_add(u):
                    ctx.add(u)
                    Sset.add(u)
                    unused.remove(u)
                    improved = True
                    added = True
                    break

            if len(ctx.S) >= MAXC:
                break

            if not added:
                # Try swaps
                S_sorted = sorted(ctx.S, key=lambda x: pair_conflict[x], reverse=True)
                swapped = False
                max_swaps = min(len(unused), 80)

                for u in unused[:max_swaps]:
                    if time.time() > deadline - 3:
                        break
                    for s in S_sorted:
                        # Try removing s and adding u
                        old_S = ctx.S[:]
                        ctx.remove(s)
                        if ctx.can_add(u):
                            ctx.add(u)
                            Sset.remove(s)
                            Sset.add(u)
                            unused.remove(u)
                            unused.append(s)
                            unused.sort(key=lambda x: pair_conflict[x])
                            improved = True
                            swapped = True
                            break
                        else:
                            # Revert
                            ctx.S = old_S[:]
                            # Rebuild DP
                            ctx.n = len(ctx.S)
                            ctx.maxsum = sum(ctx.S)
                            ctx.dp_full = build_dp(ctx.S)
                            ctx.dp_without = {}
                            for a in ctx.S:
                                sub = [y for y in ctx.S if y != a]
                                ctx.dp_without[a] = (build_dp(sub), ctx.maxsum - a)

                    if swapped:
                        break

                if not swapped:
                    no_improve += 1
                    if no_improve > 12:
                        break

        return ctx

    # Phase 1: Multiple greedy restarts
    candidates = list(range(1, N + 1))
    order1 = sorted(candidates, key=lambda x: (pair_conflict[x], deg[x], random.random()))
    order2 = sorted(candidates, key=lambda x: (deg[x], pair_conflict[x], random.random()))

    num_restarts = 60

    for i in range(num_restarts):
        if time.time() > deadline - 25:
            break

        if i % 4 == 0:
            order = order1[:]
        elif i % 4 == 1:
            order = order2[:]
        elif i % 4 == 2:
            order = candidates[:]
            random.shuffle(order)
        else:
            # Weighted random: prefer low pair_conflict
            weights = [1.0 / (pair_conflict[x] + 1) for x in candidates]
            total = sum(weights)
            order = []
            pool = candidates[:]
            for _ in range(len(pool)):
                if time.time() > deadline - 25:
                    break
                r = random.random() * total
                cum = 0
                for j, x in enumerate(pool):
                    cum += weights[pool.index(x)] if pool.index(x) < len(weights) else 0
                    if r <= cum:
                        order.append(x)
                        pool.remove(x)
                        total -= weights[pool.index(x)] if pool.index(x) < len(weights) else 0
                        break
                else:
                    if pool:
                        order.append(pool.pop())
            break

        ctx = greedy_build(order)
        if len(ctx.S) > best_sz:
            best_sz = len(ctx.S)
            best = ctx.S[:]

        if best_sz >= MAXC:
            break

    # Phase 2: Local search on best
    if best_sz < MAXC and time.time() < deadline - 10:
        ctx = DPContext()
        for x in best:
            ctx.add(x)
        ctx = local_search(ctx)
        if verify(ctx.S) and len(ctx.S) > best_sz:
            best_sz = len(ctx.S)
            best = ctx.S[:]

    # Phase 3: More local search attempts
    if best_sz < MAXC and time.time() < deadline - 5:
        for i in range(8):
            if time.time() > deadline - 5:
                break
            order = candidates[:]
            random.shuffle(order)
            ctx = greedy_build(order)
            if len(ctx.S) >= best_sz - 3:
                ctx = local_search(ctx)
                if verify(ctx.S) and len(ctx.S) > best_sz:
                    best_sz = len(ctx.S)
                    best = ctx.S[:]
            if best_sz >= MAXC:
                break

    # Final safety check
    if not verify(best):
        for sz in range(len(best), 0, -1):
            if verify(best[:sz]):
                best = best[:sz]
                break
        else:
            best = []

    return best
