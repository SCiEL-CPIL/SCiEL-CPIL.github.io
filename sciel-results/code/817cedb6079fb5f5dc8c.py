import os, sys, inspect, importlib, pkgutil, traceback
try:
    import autoresearch_suite as A
    print("PKG FILE:", A.__file__)
    print("SUBMODULES:", [m.name for m in pkgutil.iter_modules(A.__path__)])
except Exception as e:
    print("no autoresearch_suite:", e)
try:
    import autoresearch_suite.science_tcs_expansion as S
    print("MODULE FILE:", S.__file__)
    print("NAMES:", [n for n in dir(S) if not n.startswith('__')])
    for name in ("pebble_baseline",):
        try:
            f = getattr(S, name)
            print("----", name, inspect.getsource(f))
        except Exception as e:
            print("no source for", name, e)
except Exception as e:
    print("no module:", e)
