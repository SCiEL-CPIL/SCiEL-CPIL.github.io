  def _find_c6(adj, n):
      import random
      for _ in range(1000):
          u = random.randint(0, n-1)
          nu = np.where(adj[u])[0]
          if len(nu) < 2: continue
          a = random.choice(nu)
          na = np.where(adj[a])[0]
          na = [x for x in na if x != u]
          if not na: continue
          b = random.choice(na)
          nb = np.where(adj[b])[0]
          nb = [x for x in nb if x not in (u, a)]
          if not nb: continue
          c = random.choice(nb)
          nc = np.where(adj[c])[0]
          nc = [x for x in nc if x not in (u, a, b)]
          if not nc: continue
          d = random.choice(nc)
          nd = np.where(adj[d])[0]
          nd = [x for x in nd if x not in (u, a, b, c)]
          # We need d connected to e, and e connected to u.
          # e must be in N(u) \ {a}
          # e must be in N(d)
          # e must not be in {a,b,c,d}
          # e must be distinct from u
          
          nu2 = np.where(adj[u])[0]
          nu2 = [x for x in nu2 if x != a]
          common = set(nd) & set(nu2)
          common.discard(u)
          common.discard(a)
          common.discard(b)
          common.discard(c)
          common.discard(d)
          if common:
              e = random.choice(list(common))
              # Check distinctness: u, a, b, c, d, e
              if len({u, a, b, c, d, e}) == 6:
                  return [u, a, b, c, d, e]
      return None
  