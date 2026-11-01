# A = [[a, b], [c, d]], B = [[e, f], [g, h]]
# M1 = (a+d)*(e+h)
# M2 = (c+d)*e
# M3 = a*(f-h)
# M4 = d*(g-e)
# M5 = (a+b)*h
# M6 = (c-a)*(e+g)
# M7 = (b-d)*(g+h)
#
# C00 = M1 - M4 - M5 + M7  <-- WAIT, I need to be sure.
# C11 = M1 - M2 + M4 + M7
# C01 = M3 + M5
# C10 = M2 + M4
