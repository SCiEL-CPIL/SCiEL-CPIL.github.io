<<<<<<< SEARCH
        # IC constraints: p[tp] - p[t] >= v1t*(q1[tp]-q1[t]) + v2t*(q2[tp]-q2[t])
        # => -p[tp] + p[t] <= -(v1t*(q1[tp]-q1[t]) + v2t*(q2[tp]-q2[t]))
        for t in range(n):
            v1t, v2t = types[t]
            for tp in range(n):
                if tp == t:
                    continue
                row = np.zeros(n)
                row[t] = 1.0
                row[tp] = -1.0
                A_ub_p.append(row)
                b_ub_p.append(-(v1t * (q1[tp] - q1[t]) + v2t * (q2[tp] - q2[t])))
=======
        # IC constraints: adjacent types only (sufficient for lattice)
        for v1 in range(3):
            for v2 in range(4):
                t = v1 * 4 + v2
                tp = (v1 + 1) * 4 + v2
                v1t, v2t = types[t]
                row = np.zeros(n)
                row[t] = 1.0
                row[tp] = -1.0
                A_ub_p.append(row)
                b_ub_p.append(-(v1t * (q1[tp] - q1[t]) + v2t * (q2[tp] - q2[t])))

        for v1 in range(4):
            for v2 in range(3):
                t = v1 * 4 + v2
                tp = v1 * 4 + (v2 + 1)
                v1t, v2t = types[t]
                row = np.zeros(n)
                row[t] = 1.0
                row[tp] = -1.0
                A_ub_p.append(row)
                b_ub_p.append(-(v1t * (q1[tp] - q1[t]) + v2t * (q2[tp] - q2[t])))
>>>>>>> REPLACE

