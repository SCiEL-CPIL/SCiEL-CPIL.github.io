# 2x2*2x2 rank-7 terms (for matrices X,Y -> Z = X*Y, flattened row-major):
# X = [[x0,x1],[x2,x3]], Y = [[y0,y1],[y2,y3]], Z = [[z0,z1],[z2,z3]]
# z0 = M1+M4-M5+M7, z1 = M3+M5, z2 = M2+M4, z3 = M1-M2+M3+M6
# M1=(x0+x3)(y0+y3), M2=(x2+x3)(y0), M3=x0*(y1-y3)
# M4=x3*(y2-y0), M5=(x0+x1)(y3), M6=(x2-x0)(y0+y1), M7=(x1-x3)(y2+y3)
