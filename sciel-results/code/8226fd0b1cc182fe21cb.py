import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification (exact, bounded) ----------------
    def check_girth(n, edges, g, time_limit):
        if n == 0:
            return -1
        adj = [[] for _ in range(n)]
        seen = set()
        for i, j in edges:
            if i == j:
                return -1
            key = (i, j) if i < j else (j, i)
            if key in seen:
                return -1
            seen.add(key)
            adj[i].append(j)
            adj[j].append(i)
        for i in range(n):
            if len(adj[i]) != k:
                return -1
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                dn = dist[node]
                pn = parent[node]
                for nb in adj[node]:
                    if nb == pn:
                        continue
                    if dist[nb] != -1:
                        cl = dn + dist[nb] + 1
                        if cl < min_cycle:
                            min_cycle = cl
                            if cl < g:
                                return min_cycle
                    else:
                        dist[nb] = dn + 1
                        parent[nb] = node
                        queue.append(nb)
        return min_cycle

    # ---------------- base graph helpers ----------------
    def random_cubic(n, min_girth, rng, tries=400):
        if n % 2 != 0:
            return None
        for _ in range(tries):
            adj = [[] for _ in range(n)]
            for i in range(n):
                perm = list(range(n))
                rng.shuffle(perm)
                for j in perm:
                    if i == j:
                        continue
                    if len(adj[i]) >= 3:
                        break
                    if j not in adj[i]:
                        adj[i].append(j)
            if any(len(adj[i]) != 3 for i in range(n)):
                continue
            # symmetrize
            ok = True
            for i in range(n):
                for j in adj[i]:
                    if i not in adj[j]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # girth check
            gc = check_girth(n, [(min(i, j), max(i, j)) for i in range(n)
                                 for j in adj[i] if i < j], min_girth, time.time() + 0.5)
            if gc >= min_girth:
                return adj
        return None

    def base_graphs(k, g):
        bases = []
        # Petersen (girth 5)
        pet = [
            [1, 4, 5], [0, 2, 6], [1, 3, 7], [2, 4, 8], [3, 0, 9],
            [4, 6, 7], [5, 7, 8], [6, 8, 9], [7, 9, 5], [8, 6, 4],
        ]
        bases.append(pet)
        rng = random.Random(12345)
        # a few random girth-5 cubic graphs (girth >=5 keeps cycles short => good constraints)
        for n in (12, 14, 16, 18):
            for _ in range(3):
                adj = random_cubic(n, 5, rng)
                if adj is not None:
                    bases.append(adj)
                    break
        return bases

    # ---------------- walk constraint generation ----------------
    def gen_walks(base, g, L):
        n = len(base)
        adj = base
        edge_id = {}
        eid = 0
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_id[(u, v)] = eid
                    eid += 1
        E = eid
        vecs = set()
        for s in range(n):
            stack = [(s, -1, 0, np.zeros(E, dtype=np.int8))]
            while stack:
                cur, prev, length, cnt = stack.pop()
                if length >= L:
                    continue
                for nb in adj[cur]:
                    if nb == prev:
                        continue
                # (loop below)
                for nb in adj[cur]:
                    if nb == prev:
                        continue
                    key = (cur, nb) if cur < nb else (nb, cur)
                    ei = edge_id[key]
                    new_cnt = cnt.copy()
                    new_cnt[ei] += 1
                    nl = length + 1
                    if nb == s and nl >= 5:
                        v = new_cnt
                        if np.any(v != 0):
                            nz = np.nonzero(v)[0][0]
                            if v[nz] < 0:
                                v = -v
                            vecs.add(v.tobytes())
                    stack.append((nb, cur, nl, new_cnt))
        if not vecs:
            return None, E
        C = np.frombuffer(b''.join(sorted(vecs)), dtype=np.int8).reshape(-1, E).astype(np.int64)
        return C, E

    # ---------------- voltage feasibility solver ----------------
    def solve_voltage(C, E, m, rng, time_limit):
        v = rng.randint(1, m - 1, size=E).astype(np.int64)
        cost = int(np.sum(np.mod(C @ v, m) == 0))
        best_v = v.copy()
        best_cost = cost
        if best_cost == 0:
            return best_v
        # fast Gauss-Seidel: for each violated constraint, pick a random
        # non-zero coordinate and set it to cancel the constraint.
        t0 = time.time()
        stall = 0
        while best_cost > 0 and time.time() - t0 < time_limit * 0.7 and stall < 6000:
            resid = np.mod(C @ v, m)
            bad = np.nonzero(resid == 0)[0]
            if len(bad) == 0:
                break
            r = bad[rng.randrange(len(bad))]
            row = C[r]
            nz = np.nonzero(row)[0]
            if len(nz) == 0:
                stall += 1
                continue
            idx = nz[rng.randrange(len(nz))]
            a = int(row[idx])
            cur = int(v[idx])
            # choose new value so (sum + (new-cur)*a) % m != 0
            base_res = int(resid[r])  # currently 0
            # new resid = (0 + (new-cur)*a) % m ; pick new so this != 0
            inv = pow(a, -1, m)
            # want (new - cur)*a % m != 0  =>  new != cur (mod m)
            new = cur + 1 + rng.randrange(m - 1)
            new %= m
            if new == 0:
                new = 1
            v[idx] = new
            cost = int(np.sum(np.mod(C @ v, m) == 0))
            if cost < best_cost:
                best_cost = cost
                best_v = v.copy()
                stall = 0
            else:
                stall += 1
                v[idx] = cur
        # anneal polish on best
        if best_cost > 0 and time.time() - t0 < time_limit:
            v = best_v.copy()
            T = 1.0
            t1 = time.time()
            while best_cost > 0 and time.time() - t1 < (time_limit - (time.time() - t0)):
                j = rng.randrange(E)
                old = v[j]
                v[j] = rng.randint(1, m - 1)
                c = int(np.sum(np.mod(C @ v, m) == 0))
                if c < best_cost or rng.random() < np.exp(-(c - best_cost) / max(T, 1e-3)):
                    if c < best_cost:
                        best_cost = c
                        best_v = v.copy()
                else:
                    v[j] = old
                T *= 0.999
        return best_v if best_cost == 0 else None

    def build_lift(base, E, vol, m):
        n = len(base)
        edges = []
        for u in range(n):
            for vv in base[u]:
                if u < vv:
                    key = (u, vv)
                    # find edge index
                    eidx = None
                    cnt = 0
                    for a in range(n):
                        for b in base[a]:
                            if a < b:
                                if (a, b) == key:
                                    eidx = cnt
                                    break
                                cnt += 1
                        if eidx is not None:
                            break
                    volu = vol[eidx]
                    for layer in range(m):
                        u_lift = u * m + layer
                        v_lift = vv * m + (layer + volu) % m
                        if u_lift < v_lift:
                            edges.append((u_lift, v_lift))
                        else:
                            edges.append((v_lift, u_lift))
        return n * m, edges

    # ---------------- main search ----------------
    if k != 3 or g != 16:
        # generic fallback: random cubic + verify (best effort)
        rng = random.Random(999)
        for n in range(2 * k, 400, 2):
            if remaining() < 20:
                break
            for _ in range(50):
                adj = random_cubic(n, g, rng)
                if adj is not None:
                    edges = [(min(i, j), max(i, j)) for i in range(n) for j in adj[i] if i < j]
                    return (n, edges)
        return (0, [])

    L = g - 1  # 15
    bases = base_graphs(k, g)
    rng = random.Random(20240601)

    # candidate moduli (odd, product of small primes, to keep n moderate)
    moduli = [77, 75, 91, 85, 95, 105, 99, 87, 111, 115, 93, 117, 119, 121, 123]
    moduli = [m for m in moduli if m >= 10]

    for base in bases:
        for m in moduli:
            if remaining() < 15:
                break
            n_base = len(base)
            if n_base * m > 900:
                continue
            C, E = gen_walks(base, g, L)
            if C is None:
                continue
            # quick feasibility: number of constraints vs freedom
            t_lim = min(8.0, remaining() * 0.5)
            vol = solve_voltage(C, E, m, rng, t_lim)
            if vol is not None:
                n, edges = build_lift(base, E, vol, m)
                gc = check_girth(n, edges, g, deadline - 3)
                if gc >= g:
                    return (n, edges)

    # fallback: Petersen with larger modulus and more time
    base = bases[0]
    for m in [135, 145, 155, 165, 175, 185, 195, 205]:
        if remaining() < 15:
            break
        C, E = gen_walks(base, g, L)
        if C is None:
            continue
        vol = solve_voltage(C, E, m, rng, min(10.0, remaining() * 0.5))
        if vol is not None:
            n, edges = build_lift(base, E, vol, m)
            gc = check_girth(n, edges, g, deadline - 3)
            if gc >= g:
                return (n, edges)

    return (0, [])


if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
