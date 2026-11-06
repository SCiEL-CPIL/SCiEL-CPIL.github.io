def _quartet_vote(sequences, labels, n, L, max_quartets=None):
    base_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    taxon = []
    for i in range(n):
        seq = sequences[labels[i]]
        taxon.append([base_idx[c] for c in seq])
    votes = {}
    quarts = list(itertools.combinations(range(n), 4))
    if max_quartets and len(quarts) > max_quartets:
        random.seed(0)
        random.shuffle(quarts)
        quarts = quarts[:max_quartets]

    # Precompute per-quartet pattern counts lazily
    def pattern_counts(t1, t2, t3, t4):
        counts = [0] * 256
        for m in range(L):
            pat = t1[m] * 64 + t2[m] * 16 + t3[m] * 4 + t4[m]
            counts[pat] += 1
        return counts

    # Precompute transition tables for a grid of branch lengths
    # For each e (leaf branch) and l5 (internal), compute for each of the 16
    # patterns the log contribution... but likelihood is a sum, so we need the
    # unnormalized probability per pattern then sum.
    # For split (ab|cd): P(pattern) = sum_{r,p,q} (1/4) P(r->p)P(r->q)
    #   P(p->a)P(p->b) P(q->c)P(q->d)
    # Precompute for given e,l5: for each base pair (x,y) the value
    #   w_pair(x,y) = sum_p P(p->x;e)P(p->y;e)  (this is the "pair weight")
    # and w_root(p,q) = sum_r (1/4) P(r->p;l5)P(r->q;l5)
    # Then P(pattern abcd) = sum_{p,q} w_pair(a,b) ... no, p is shared.
    # Actually P = sum_{p,q} [P(p->a)P(p->b)] [P(q->c)P(q->d)] w_root(p,q)
    # = sum_{p,q} A_p B_q w_root(p,q) where A_p=P(p->a)P(p->b), B_q=P(q->c)P(q->d)
    def ll_split_counts(counts, e, l5):
        pe_s = 0.25 + 0.75 * math.exp(-e)
        pe_d = (1 - pe_s) / 3.0
        p5_s = 0.25 + 0.75 * math.exp(-l5)
        p5_d = (1 - p5_s) / 3.0
        # A_p for given (a,b): but a,b vary per pattern. We'll compute per pattern.
        total = 0.0
        for pat in range(256):
            cnt = counts[pat]
            if cnt == 0:
                continue
            a = pat >> 6
            b = (pat >> 4) & 3
            c = (pat >> 2) & 3
            d = pat & 3
            # A_p = P(p->a)*P(p->b)
            A = [0.0] * 4
            for p in range(4):
                A[p] = (pe_s if p == a else pe_d) * (pe_s if p == b else pe_d)
            B = [0.0] * 4
            for q in range(4):
                B[q] = (pe_s if q == c else pe_d) * (pe_s if q == d else pe_d)
            # sum_{p,q} A_p B_q w_root(p,q)
            s = 0.0
            for p in range(4):
                for q in range(4):
                    w = p5_s if (p == q) else p5_d  # w_root(p,q) = sum_r (1/4)P(r->p)P(r->q)
                    # Actually w_root(p,q) = sum_r (1/4) P(r->p;l5) P(r->q;l5)
                    # For JC: if p==q: sum_r (1/4) P(r->p)^2
                    #   = (1/4)[P(same->same)^2 *1 (r=p) + P(diff->p)^2 *3]
                    #   = (1/4)[p5_s^2 + 3 p5_d^2]
                    # if p!=q: sum_r (1/4) P(r->p)P(r->q)
                    #   = (1/4)[P(p->p)P(p->q) + P(q->p)P(q->q) + sum_{r!=p,q} P(r->p)P(r->q)]
                    #   = (1/4)[p5_s*p5_d + p5_d*p5_s + 2*p5_d*p5_d]
                    #   = (1/4)[2 p5_s p5_d + 2 p5_d^2]
                    if p == q:
                        w = 0.25 * (p5_s * p5_s + 3 * p5_d * p5_d)
                    else:
                        w = 0.25 * (2 * p5_s * p5_d + 2 * p5_d * p5_d)
                    s += A[p] * B[q] * w
            total += cnt * s
        return math.log(total) if total > 0 else -1e18

    grid_e = (0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0)
    grid_l5 = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0)

    for (a, b, c, d) in quarts:
        if _remaining() < 8:
            break
        counts = pattern_counts(taxon[a], taxon[b], taxon[c], taxon[d])
        # For each of the 3 splits, find best (e,l5)
        def best_ll(t1, t2, t3, t4):
            # recompute counts in the order (t1,t2,t3,t4)
            cnt = [0] * 256
            for m in range(L):
                pat = t1[m] * 64 + t2[m] * 16 + t3[m] * 4 + t4[m]
                cnt[pat] += 1
            best = -1e18
            for e in grid_e:
                for l5 in grid_l5:
                    v = ll_split_counts(cnt, e, l5)
                    if v > best:
                        best = v
            return best
        ta, tb, tc, td = taxon[a], taxon[b], taxon[c], taxon[d]
        ll1 = best_ll(ta, tb, tc, td)
        ll2 = best_ll(ta, tc, tb, td)
        ll3 = best_ll(ta, td, tb, tc)
        best = max(ll1, ll2, ll3)
        key = frozenset((a, b, c, d))
        if ll1 == best:
            votes[key] = (frozenset((a, b)), frozenset((c, d)))
        elif ll2 == best:
            votes[key] = (frozenset((a, c)), frozenset((b, d)))
        else:
            votes[key] = (frozenset((a, d)), frozenset((b, c)))
    return votes
