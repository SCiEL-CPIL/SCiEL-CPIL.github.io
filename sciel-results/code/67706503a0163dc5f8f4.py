i_new = len(family)
for i_old in range(i_new):
    inter = popcount(family[i_old] & m)
    pair_intersections[(i_old, i_new)] = inter
family.append(m)
